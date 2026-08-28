"""Static checks for tiered Selena synchronization and Git publication."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TransferTierContractTest(unittest.TestCase):
    def test_result_sync_defaults_to_lightweight_and_offers_detail(self) -> None:
        sync = (ROOT / "sync_results_to_dgx.sh").read_text(encoding="utf-8")

        self.assertIn('SYNC_SIZE="lightweight"', sync)
        self.assertIn('lightweight|detailed|full)', sync)
        self.assertIn("--include=/reports/", sync)
        self.assertIn("--include=/reports/***", sync)
        self.assertIn("--exclude=/reports/**/plots/***", sync)
        self.assertIn("--exclude=/reports/**/averaged_inputs/***", sync)
        lightweight = sync.split('if [ "$SYNC_SIZE" = lightweight ]; then', 1)[1].split(
            'elif [ "$SYNC_SIZE" = detailed ]; then', 1
        )[0]
        self.assertNotIn("--include=*/", lightweight)
        self.assertNotIn("--include=*.json", lightweight)
        self.assertNotIn("--include=*.csv", lightweight)
        self.assertIn('"${OUTPUT_FILTERS[@]}"', sync)
        self.assertIn("--job-id", sync)
        self.assertIn('"--include=*_${JOB_ID}.out"', sync)
        self.assertIn('"--include=*_${JOB_ID}.err"', sync)
        self.assertNotIn("--delete", sync)

    def test_publisher_defaults_to_lightweight_and_never_publishes_caches(self) -> None:
        publisher = (ROOT / "publish_job.sh").read_text(encoding="utf-8")

        self.assertIn('publish_size="lightweight"', publisher)
        self.assertIn('lightweight|detailed)', publisher)
        self.assertIn('paths+=("$output_tree/reports")', publisher)
        self.assertIn('if [ "$publish_size" = detailed ]', publisher)
        self.assertNotIn("**/window_metrics.csv", publisher)
        self.assertNotIn("**/per_user_date_metrics.csv", publisher)
        self.assertIn("**/*.pt", publisher)
        self.assertIn("**/*.npy", publisher)
        self.assertIn("**/*.cbm", publisher)


if __name__ == "__main__":
    unittest.main()
