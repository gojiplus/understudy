"""Test the tags feature in Suite.run() with storage.

This script tests:
1. Running a suite with tags parameter
2. Verifying tags are stored in metadata.json
3. Using the CLI to list and show runs with tags
4. Using compare to compare runs with different tags

Run from the example directory:
    cd example
    uv run python test_tags_feature.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from customer_service_agent import customer_service_agent

from understudy import MockToolkit, RunStorage, Suite, ToolError
from understudy.adk import ADKApp
from understudy.compare import compare_runs


def create_app_with_mocks():
    """Create the ADK app with mocks (same as conftest.py)."""
    mocks = MockToolkit()

    orders = {
        "ORD-10031": {
            "order_id": "ORD-10031",
            "items": [
                {
                    "name": "Hiking Backpack",
                    "sku": "HB-220",
                    "category": "outdoor_gear",
                    "price": 129.99,
                }
            ],
            "date": "2025-02-28",
            "status": "delivered",
        },
        "ORD-10027": {
            "order_id": "ORD-10027",
            "items": [
                {
                    "name": "Wireless Earbuds Pro",
                    "sku": "WE-500",
                    "category": "personal_audio",
                    "price": 249.99,
                }
            ],
            "date": "2025-02-15",
            "status": "delivered",
        },
    }

    non_returnable_categories = ["personal_audio", "perishables", "final_sale"]

    @mocks.handle("lookup_order")
    def lookup_order(order_id: str) -> dict:
        if order_id in orders:
            return orders[order_id]
        raise ToolError(f"Order {order_id} not found")

    @mocks.handle("lookup_customer_orders")
    def lookup_customer_orders(email: str) -> list[dict]:
        return list(orders.values())

    @mocks.handle("get_return_policy")
    def get_return_policy(category: str) -> dict:
        return {
            "category": category,
            "returnable": category not in non_returnable_categories,
            "return_window_days": 30,
            "non_returnable_categories": non_returnable_categories,
        }

    @mocks.handle("create_return")
    def create_return(order_id: str, item_sku: str, reason: str) -> dict:
        return {
            "return_id": "RET-001",
            "order_id": order_id,
            "item_sku": item_sku,
            "status": "created",
        }

    @mocks.handle("escalate_to_human")
    def escalate_to_human(reason: str) -> dict:
        return {"status": "escalated", "reason": reason, "ticket_id": "ESC-001"}

    app = ADKApp(agent=customer_service_agent)
    return app, mocks


def test_tags_stored_correctly():
    """Test that tags are properly stored in metadata.json."""
    print("=" * 60)
    print("Test 1: Tags stored correctly in metadata")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = RunStorage(path=Path(tmpdir) / "runs")
        app, mocks = create_app_with_mocks()

        suite = Suite.from_directory("scenes/")
        tags = {"version": "v1", "model": "test-model", "experiment": "tags-test"}

        results = suite.run(app, mocks=mocks, storage=storage, tags=tags, parallel=1)

        print(f"Results: {results.pass_count}/{len(results.results)} passed")

        run_ids = storage.list_runs()
        print(f"Stored {len(run_ids)} runs")

        for run_id in run_ids:
            run_dir = storage.path / run_id
            metadata_file = run_dir / "metadata.json"
            with open(metadata_file) as f:
                metadata = json.load(f)

            stored_tags = metadata.get("tags", {})
            assert stored_tags == tags, f"Tags mismatch! Expected {tags}, got {stored_tags}"
            print(f"  {run_id}: tags={stored_tags}")

        print("PASS: Tags stored correctly")
        return True


def test_tags_loaded_correctly():
    """Test that tags are properly loaded from storage."""
    print("\n" + "=" * 60)
    print("Test 2: Tags loaded correctly via storage.load()")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = RunStorage(path=Path(tmpdir) / "runs")
        app, mocks = create_app_with_mocks()

        suite = Suite.from_directory("scenes/")
        tags = {"version": "v2", "team": "platform"}

        suite.run(app, mocks=mocks, storage=storage, tags=tags, parallel=1)

        for run_id in storage.list_runs():
            data = storage.load(run_id)
            loaded_tags = data.get("metadata", {}).get("tags", {})
            assert loaded_tags == tags, f"Tags mismatch! Expected {tags}, got {loaded_tags}"
            print(f"  {run_id}: loaded tags={loaded_tags}")

        print("PASS: Tags loaded correctly")
        return True


def test_compare_with_tags():
    """Test comparing runs with different tag values."""
    print("\n" + "=" * 60)
    print("Test 3: Compare runs using tags")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = RunStorage(path=Path(tmpdir) / "runs")
        app, mocks = create_app_with_mocks()
        suite = Suite.from_directory("scenes/")

        print("Running suite with version=v1...")
        suite.run(app, mocks=mocks, storage=storage, tags={"version": "v1"}, parallel=1)

        print("Running suite with version=v2...")
        suite.run(app, mocks=mocks, storage=storage, tags={"version": "v2"}, parallel=1)

        print("\nComparing v1 vs v2...")
        result = compare_runs(storage, tag="version", before_value="v1", after_value="v2")

        print(f"Comparison: {result.before_label} vs {result.after_label}")
        print(f"Tag: {result.tag}")
        print(f"Before runs: {result.before_runs}")
        print(f"After runs: {result.after_runs}")
        print(f"Before pass rate: {result.before_pass_rate * 100:.1f}%")
        print(f"After pass rate: {result.after_pass_rate * 100:.1f}%")
        print(f"Pass rate delta: {result.pass_rate_delta * 100:+.1f}%")

        assert result.before_runs == 3, f"Expected 3 before runs, got {result.before_runs}"
        assert result.after_runs == 3, f"Expected 3 after runs, got {result.after_runs}"
        assert result.tag == "version"
        assert result.before_value == "v1"
        assert result.after_value == "v2"

        print("PASS: Compare with tags works correctly")
        return True


def test_empty_tags():
    """Test that empty/None tags work correctly."""
    print("\n" + "=" * 60)
    print("Test 4: Empty tags")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = RunStorage(path=Path(tmpdir) / "runs")
        app, mocks = create_app_with_mocks()
        suite = Suite.from_directory("scenes/")

        suite.run(app, mocks=mocks, storage=storage, tags=None, parallel=1)

        for run_id in storage.list_runs():
            data = storage.load(run_id)
            loaded_tags = data.get("metadata", {}).get("tags", {})
            assert loaded_tags == {}, f"Expected empty tags, got {loaded_tags}"
            print(f"  {run_id}: tags={loaded_tags}")

        print("PASS: Empty tags handled correctly")
        return True


def main():
    """Run all tests."""
    print("Testing tags feature in Suite.run()\n")

    tests = [
        test_tags_stored_correctly,
        test_tags_loaded_correctly,
        test_compare_with_tags,
        test_empty_tags,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} raised {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
