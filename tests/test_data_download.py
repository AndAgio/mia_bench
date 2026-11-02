"""Regressions for python -m src.data, with no downloads."""
import io
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.data.__main__ as download
from src.data import News, Texas


def mocked_downloads():
    # Every dataset with a mock class, and the same splits and options as the real entry.
    return {name: (printed, MagicMock(name=name), splits, options)
            for name, (printed, _, splits, options) in download.DOWNLOADS.items()}


class DownloadTest(unittest.TestCase):
    def run_main(self, *argv):
        with redirect_stdout(io.StringIO()):
            download.main(list(argv))

    def test_texas_and_news_download_both_splits(self):
        for name, dataset_class in (('texas', Texas), ('news', News)):
            self.assertEqual(download.DOWNLOADS[name][1:], (dataset_class, download.TRAIN_TEST, {'download': True}))

    def test_named_datasets_only(self):
        downloads = mocked_downloads()
        with patch.dict(download.DOWNLOADS, downloads):
            self.run_main('texas', 'news', '--datasets_folder', '/data')
        for name, (_, dataset_class, _, _) in downloads.items():
            if name in ('texas', 'news'):
                self.assertEqual(dataset_class.call_args_list,
                                 [call(root=f'/data/{name}', transform=None, train=train, download=True)
                                  for train in (True, False)])
            else:
                dataset_class.assert_not_called()

    def test_all_datasets_without_names(self):
        downloads = mocked_downloads()
        with patch.dict(download.DOWNLOADS, downloads):
            self.run_main('--datasets_folder', '/data')
        self.assertTrue(all(dataset_class.call_count == 2 for _, dataset_class, _, _ in downloads.values()))

    def test_unknown_dataset_is_an_error(self):
        with patch.dict(download.DOWNLOADS, mocked_downloads()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.run_main('texas', 'mnist')


if __name__ == '__main__':
    unittest.main()
