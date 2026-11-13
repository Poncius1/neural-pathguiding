"""Static regression for the source-preserving extraction migration.

Does not require Mitsuba or a trained checkpoint.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.apply_direct_estimator import (
    DEST, DEPENDENTS, MOVED, NEW_IMPORT, OLD_IMPORT,
    NEW_MODULE_IMPORT, OLD_MODULE_IMPORT, SOURCE, plan,
)


class TestDirectEstimatorRefactor(unittest.TestCase):
    def test_preserves_function_text_and_updates_consumers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / SOURCE).parent.mkdir(parents=True)
            functions = ''.join(f'def {name}():\n    return {i}\n\n' for i, name in enumerate((*MOVED, 'parse_arguments', 'run_estimator')))
            original = functions + 'def main():\n    return True\n\nif __name__ == "__main__":\n    main()\n'
            (root / SOURCE).write_text(original, encoding='utf-8')
            for rel in DEPENDENTS:
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(OLD_IMPORT + '\n    sample_fixed_surface,\n)\n', encoding='utf-8')
            changes = plan(root)
            self.assertEqual(len(changes), len(DEPENDENTS) + 2)
            extracted = changes[root / DEST]
            for i, name in enumerate(MOVED):
                self.assertIn(f'def {name}():\n    return {i}', extracted)
                self.assertNotIn(f'def {name}():\n    return {i}', changes[root / SOURCE])
            self.assertIn('def main():', changes[root / SOURCE])
            self.assertIn(NEW_IMPORT, changes[root / DEPENDENTS[0]])
            # The planning phase never changes the original files.
            self.assertEqual((root / SOURCE).read_text(encoding='utf-8'), original)

    def test_single_line_legacy_import_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / SOURCE).parent.mkdir(parents=True)
            functions = ''.join(
                f'def {name}():\n    return {i}\n\n'
                for i, name in enumerate((*MOVED, 'parse_arguments', 'run_estimator'))
            )
            (root / SOURCE).write_text(
                functions + 'def main():\n    pass\n\nif __name__ == "__main__":\n    main()\n',
                encoding='utf-8',
            )
            single_line = 'scripts/render_t2_v2_multibounce.py'
            for rel in DEPENDENTS:
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                source = (OLD_MODULE_IMPORT + 'sha256_file\n') if rel == single_line else (
                    OLD_IMPORT + '\n    sample_fixed_surface,\n)\n'
                )
                path.write_text(source, encoding='utf-8')
            changes = plan(root)
            self.assertIn(
                NEW_MODULE_IMPORT + 'sha256_file', changes[root / single_line]
            )
            self.assertEqual(
                (root / single_line).read_text(encoding='utf-8'),
                OLD_MODULE_IMPORT + 'sha256_file\n',
            )

    def test_existing_destination_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / DEST).parent.mkdir(parents=True)
            (root / DEST).write_text('already exists', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                plan(root)


if __name__ == '__main__':
    unittest.main()
