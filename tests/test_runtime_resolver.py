from copy import deepcopy
import unittest

from runtime_resolver import resolve_runtime, installed_matches


def amd_host(kernel='6.14.0-20-generic'):
    return dict(system='Linux', machine='x86_64', os_id='ubuntu', os_version='24.04',
                os_release='24.04.3', kernel=kernel, python_version='3.12.3', glibc_version='2.39',
                backend='amd', rocm_version='7.2.0', amdgpu_module_version='7.1.3.31500000',
                gpus=[dict(index=str(i), name='AMD Radeon AI Pro R9700S', architecture='gfx1201',
                           driver_version='6.16.13') for i in range(8)], installed_packages={})


def nvidia_host(driver='570.86.16', arch='sm_120'):
    host = amd_host('6.8.0-60-generic')
    host.update(backend='nvidia', gpus=[dict(index='0', name='RTX', architecture=arch, driver_version=driver)])
    return host


class RuntimeResolverTest(unittest.TestCase):
    def test_inspected_server_is_candidate_but_kernel_requires_acknowledgement(self):
        host = amd_host('6.8.0-137-generic')
        plan = resolve_runtime(host)
        self.assertEqual(plan['status'], 'blocked')
        self.assertEqual(plan['profile']['torch_version'], '2.9.1')
        self.assertTrue(any('kernel' in issue for issue in plan['errors']))
        accepted = resolve_runtime(host, allow_unverified_host=True)
        self.assertEqual(accepted['status'], 'compatible_with_warnings')
        self.assertFalse(accepted['host_qualified'])
        self.assertTrue(accepted['warnings'])

    def test_amd_uses_versioned_radeon_wheels_for_actual_python(self):
        plan = resolve_runtime(amd_host())
        self.assertEqual(plan['status'], 'compatible')
        wheels = plan['profile']['core_install']['packages']
        self.assertTrue(all('repo.radeon.com/rocm/manylinux/rocm-rel-7.2/' in url for url in wheels))
        self.assertTrue(all('cp312-cp312' in url for url in wheels))
        self.assertTrue(any('torch-2.9.1%2Brocm7.2.0' in url for url in wheels))

    def test_blackwell_cannot_fall_back_to_cuda126_on_old_driver(self):
        plan = resolve_runtime(nvidia_host('560.35.03'))
        self.assertEqual(plan['status'], 'blocked')
        self.assertTrue(any('driver' in issue for issue in plan['errors']))
        self.assertEqual(resolve_runtime(nvidia_host())['profile']['id'], 'torch291-cu128')

    def test_ada_selects_supported_older_cuda_without_changing_torch_release(self):
        plan = resolve_runtime(nvidia_host('560.35.03', 'sm_89'))
        self.assertEqual(plan['status'], 'compatible')
        self.assertEqual(plan['profile']['id'], 'torch291-cu126')
        self.assertEqual(plan['profile']['torch_version'], '2.9.1')

    def test_override_cannot_bypass_architecture_or_runtime_failure(self):
        for field, value in (('rocm_version', '6.4.3'), ('python_version', '3.11.9')):
            host = amd_host()
            host[field] = value
            with self.subTest(field=field):
                self.assertEqual(resolve_runtime(host, allow_unverified_host=True)['status'], 'blocked')
        host = amd_host()
        host['gpus'][0]['architecture'] = 'gfx9999'
        self.assertEqual(resolve_runtime(host, allow_unverified_host=True)['status'], 'blocked')
        self.assertEqual(resolve_runtime(nvidia_host(), profile='torch291-rocm72')['status'], 'blocked')

    def test_existing_wrong_backend_or_package_build_does_not_match(self):
        plan = resolve_runtime(amd_host())
        expected = deepcopy(plan['profile']['expected_packages'])
        self.assertTrue(installed_matches(plan, expected))
        expected['torch'] = '2.9.1+cu128'
        self.assertFalse(installed_matches(plan, expected))

    def test_resolution_does_not_modify_observations(self):
        host = amd_host()
        original = deepcopy(host)
        resolve_runtime(host)
        self.assertEqual(host, original)

    def test_missing_driver_or_architecture_is_not_guessed(self):
        for field in ('driver_version', 'architecture'):
            host = nvidia_host()
            host['gpus'][0][field] = None
            self.assertEqual(resolve_runtime(host)['status'], 'blocked')
