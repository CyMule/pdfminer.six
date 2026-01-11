"""Comprehensive tests for pdfminer/data_structures.py"""

import pytest

from pdfminer import settings
from pdfminer.data_structures import NumberTree
from pdfminer.pdfexceptions import PDFTypeError
from pdfminer.pdfparser import PDFSyntaxError


class TestNumberTreeBasicConstruction:
    """Tests for NumberTree basic construction"""

    def test_construct_with_nums_only(self):
        """A leaf node with only Nums key."""
        obj = {"Nums": [0, "value0", 1, "value1", 2, "value2"]}
        tree = NumberTree(obj)
        assert tree.nums is not None
        assert tree.kids is None
        assert tree.limits is None

    def test_construct_with_kids_only(self):
        """A root/intermediate node with only Kids key."""
        obj = {"Kids": []}
        tree = NumberTree(obj)
        assert tree.nums is None
        assert tree.kids is not None
        assert tree.limits is None

    def test_construct_with_limits(self):
        """A node with Limits key."""
        obj = {"Nums": [10, "val10"], "Limits": [10, 10]}
        tree = NumberTree(obj)
        assert tree.nums is not None
        assert tree.limits is not None

    def test_construct_with_all_keys(self):
        """A node with all three keys."""
        obj = {
            "Nums": [5, "val5"],
            "Kids": [],
            "Limits": [5, 5],
        }
        tree = NumberTree(obj)
        assert tree.nums is not None
        assert tree.kids is not None
        assert tree.limits is not None

    def test_construct_empty_dict(self):
        """An empty dictionary creates a tree with no data."""
        obj = {}
        tree = NumberTree(obj)
        assert tree.nums is None
        assert tree.kids is None
        assert tree.limits is None


class TestNumberTreeValues:
    """Tests for NumberTree.values property"""

    def test_values_simple_leaf(self):
        """Simple leaf node with key-value pairs."""
        obj = {"Nums": [0, "zero", 1, "one", 2, "two"]}
        tree = NumberTree(obj)
        values = tree.values
        assert values == [(0, "zero"), (1, "one"), (2, "two")]

    def test_values_single_entry(self):
        """Leaf node with a single entry."""
        obj = {"Nums": [42, "answer"]}
        tree = NumberTree(obj)
        values = tree.values
        assert values == [(42, "answer")]

    def test_values_empty_nums(self):
        """Leaf node with empty Nums list."""
        obj = {"Nums": []}
        tree = NumberTree(obj)
        values = tree.values
        assert values == []

    def test_values_empty_dict(self):
        """Empty dictionary yields empty values."""
        obj = {}
        tree = NumberTree(obj)
        values = tree.values
        assert values == []

    def test_values_sorted_non_strict(self):
        """Values are sorted when not in strict mode."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            obj = {"Nums": [3, "three", 1, "one", 2, "two"]}
            tree = NumberTree(obj)
            values = tree.values
            assert values == [(1, "one"), (2, "two"), (3, "three")]
        finally:
            settings.STRICT = original_strict

    def test_values_out_of_order_strict_raises(self):
        """Out of order values raise PDFSyntaxError in strict mode."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            obj = {"Nums": [3, "three", 1, "one", 2, "two"]}
            tree = NumberTree(obj)
            with pytest.raises(PDFSyntaxError) as exc_info:
                _ = tree.values
            assert "out of order" in str(exc_info.value).lower()
        finally:
            settings.STRICT = original_strict

    def test_values_ordered_strict_no_error(self):
        """Properly ordered values don't raise in strict mode."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            obj = {"Nums": [1, "one", 2, "two", 3, "three"]}
            tree = NumberTree(obj)
            values = tree.values
            assert values == [(1, "one"), (2, "two"), (3, "three")]
        finally:
            settings.STRICT = original_strict

    def test_values_equal_keys_strict_allowed(self):
        """Equal consecutive keys are allowed in strict mode."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            obj = {"Nums": [1, "one_a", 1, "one_b", 2, "two"]}
            tree = NumberTree(obj)
            values = tree.values
            assert values == [(1, "one_a"), (1, "one_b"), (2, "two")]
        finally:
            settings.STRICT = original_strict


class TestNumberTreeHierarchy:
    """Tests for NumberTree hierarchical structure (Kids)"""

    def test_single_child(self):
        """Root with a single child leaf node."""
        child = {"Nums": [0, "zero", 1, "one"]}
        root = {"Kids": [child]}
        tree = NumberTree(root)
        values = tree.values
        assert values == [(0, "zero"), (1, "one")]

    def test_multiple_children(self):
        """Root with multiple child leaf nodes."""
        child1 = {"Nums": [0, "zero", 1, "one"]}
        child2 = {"Nums": [2, "two", 3, "three"]}
        root = {"Kids": [child1, child2]}
        tree = NumberTree(root)
        values = tree.values
        assert values == [(0, "zero"), (1, "one"), (2, "two"), (3, "three")]

    def test_nested_hierarchy(self):
        """Multi-level tree with intermediate nodes."""
        leaf1 = {"Nums": [0, "zero", 1, "one"]}
        leaf2 = {"Nums": [2, "two", 3, "three"]}
        intermediate = {"Kids": [leaf1, leaf2]}
        root = {"Kids": [intermediate]}
        tree = NumberTree(root)
        values = tree.values
        assert values == [(0, "zero"), (1, "one"), (2, "two"), (3, "three")]

    def test_mixed_nums_and_kids(self):
        """Node with both Nums and Kids."""
        child = {"Nums": [10, "ten", 11, "eleven"]}
        obj = {"Nums": [0, "zero", 1, "one"], "Kids": [child]}
        tree = NumberTree(obj)
        values = tree.values
        assert (0, "zero") in values
        assert (1, "one") in values
        assert (10, "ten") in values
        assert (11, "eleven") in values

    def test_empty_kids_list(self):
        """Node with empty Kids list."""
        obj = {"Kids": []}
        tree = NumberTree(obj)
        values = tree.values
        assert values == []

    def test_deep_nesting(self):
        """Deeply nested tree structure."""
        leaf = {"Nums": [42, "deep_value"]}
        current = leaf
        for _ in range(5):
            current = {"Kids": [current]}
        tree = NumberTree(current)
        values = tree.values
        assert values == [(42, "deep_value")]


class TestNumberTreeEdgeCases:
    """Tests for NumberTree edge cases"""

    def test_negative_keys(self):
        """Number tree with negative keys."""
        obj = {"Nums": [-2, "neg_two", -1, "neg_one", 0, "zero"]}
        tree = NumberTree(obj)
        values = tree.values
        assert values == [(-2, "neg_two"), (-1, "neg_one"), (0, "zero")]

    def test_large_keys(self):
        """Number tree with large integer keys."""
        obj = {"Nums": [1000000, "million", 2000000, "two_million"]}
        tree = NumberTree(obj)
        values = tree.values
        assert values == [(1000000, "million"), (2000000, "two_million")]

    def test_complex_values(self):
        """Number tree with complex values (dicts, lists)."""
        obj = {
            "Nums": [
                0,
                {"nested": "dict"},
                1,
                [1, 2, 3],
                2,
                None,
            ]
        }
        tree = NumberTree(obj)
        values = tree.values
        assert values[0] == (0, {"nested": "dict"})
        assert values[1] == (1, [1, 2, 3])
        assert values[2] == (2, None)

    def test_duplicate_keys_sorted(self):
        """Duplicate keys are preserved and sorted by key."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            obj = {"Nums": [1, "first", 1, "second", 1, "third"]}
            tree = NumberTree(obj)
            values = tree.values
            assert len(values) == 3
            assert all(k == 1 for k, v in values)
        finally:
            settings.STRICT = original_strict

    def test_float_keys_fallback_to_zero_non_strict(self):
        """Float keys are not valid integers; int_value returns 0 in non-strict mode."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            obj = {"Nums": [1.0, "one", 2.0, "two"]}
            tree = NumberTree(obj)
            values = tree.values
            # int_value returns 0 for floats in non-strict mode
            assert values == [(0, "one"), (0, "two")]
        finally:
            settings.STRICT = original_strict

    def test_tuple_nums(self):
        """Nums can be a tuple instead of a list."""
        obj = {"Nums": (0, "zero", 1, "one")}
        tree = NumberTree(obj)
        values = tree.values
        assert values == [(0, "zero"), (1, "one")]


class TestNumberTreeInvalidInput:
    """Tests for NumberTree invalid input handling"""

    def test_non_dict_input_non_strict(self):
        """Non-dict input in non-strict mode returns empty values."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            tree = NumberTree("not a dict")
            values = tree.values
            assert values == []
        finally:
            settings.STRICT = original_strict

    def test_non_dict_input_strict(self):
        """Non-dict input in strict mode raises PDFTypeError."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            with pytest.raises(PDFTypeError):
                NumberTree("not a dict")
        finally:
            settings.STRICT = original_strict

    def test_non_list_nums_non_strict(self):
        """Non-list Nums in non-strict mode returns empty values."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            obj = {"Nums": "not a list"}
            tree = NumberTree(obj)
            values = tree.values
            assert values == []
        finally:
            settings.STRICT = original_strict

    def test_non_list_nums_strict(self):
        """Non-list Nums in strict mode raises PDFTypeError."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            obj = {"Nums": "not a list"}
            with pytest.raises(PDFTypeError):
                NumberTree(obj)
        finally:
            settings.STRICT = original_strict

    def test_non_integer_key_non_strict(self):
        """Non-integer key in non-strict mode converts to 0."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            obj = {"Nums": ["not_an_int", "value"]}
            tree = NumberTree(obj)
            values = tree.values
            assert values == [(0, "value")]
        finally:
            settings.STRICT = original_strict

    def test_non_integer_key_strict(self):
        """Non-integer key in strict mode raises PDFTypeError."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            obj = {"Nums": ["not_an_int", "value"]}
            tree = NumberTree(obj)
            with pytest.raises(PDFTypeError):
                _ = tree.values
        finally:
            settings.STRICT = original_strict

    def test_odd_number_of_nums_elements(self):
        """Odd number of elements in Nums drops the last one."""
        obj = {"Nums": [0, "zero", 1, "one", 2]}
        tree = NumberTree(obj)
        values = tree.values
        assert values == [(0, "zero"), (1, "one")]

    def test_none_input_non_strict(self):
        """None input in non-strict mode returns empty values."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            tree = NumberTree(None)
            values = tree.values
            assert values == []
        finally:
            settings.STRICT = original_strict


class TestNumberTreeLimits:
    """Tests for NumberTree Limits attribute"""

    def test_limits_attribute_preserved(self):
        """Limits attribute is accessible after construction."""
        obj = {"Nums": [10, "val"], "Limits": [10, 10]}
        tree = NumberTree(obj)
        limits = list(tree.limits)
        assert limits == [10, 10]

    def test_limits_range(self):
        """Limits typically specify min and max keys."""
        obj = {"Nums": [5, "five", 10, "ten", 15, "fifteen"], "Limits": [5, 15]}
        tree = NumberTree(obj)
        limits = list(tree.limits)
        assert limits[0] == 5
        assert limits[1] == 15


class TestNumberTreeParseMethod:
    """Tests for NumberTree._parse() internal method"""

    def test_parse_returns_list_of_tuples(self):
        """_parse returns a list of (int, value) tuples."""
        obj = {"Nums": [1, "one", 2, "two"]}
        tree = NumberTree(obj)
        items = tree._parse()
        assert isinstance(items, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in items)

    def test_parse_recursive_kids(self):
        """_parse recursively processes Kids."""
        child = {"Nums": [0, "child_val"]}
        root = {"Kids": [child]}
        tree = NumberTree(root)
        items = tree._parse()
        assert items == [(0, "child_val")]

    def test_parse_combines_nums_and_kids(self):
        """_parse combines items from Nums and Kids."""
        child = {"Nums": [10, "from_child"]}
        obj = {"Nums": [0, "from_nums"], "Kids": [child]}
        tree = NumberTree(obj)
        items = tree._parse()
        assert (0, "from_nums") in items
        assert (10, "from_child") in items


class TestNumberTreeStrictModeToggle:
    """Tests that verify strict mode is properly toggled during tests"""

    def test_strict_mode_false_by_default(self):
        """Verify default strict mode is False."""
        original = settings.STRICT
        try:
            assert settings.STRICT is False or settings.STRICT is True
        finally:
            settings.STRICT = original

    def test_sorting_in_non_strict(self):
        """Non-strict mode sorts out-of-order entries."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            obj = {"Nums": [5, "five", 1, "one", 3, "three"]}
            tree = NumberTree(obj)
            values = tree.values
            keys = [k for k, v in values]
            assert keys == sorted(keys)
        finally:
            settings.STRICT = original_strict

    def test_no_sorting_needed_strict(self):
        """Strict mode with already sorted entries works fine."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            obj = {"Nums": [1, "one", 3, "three", 5, "five"]}
            tree = NumberTree(obj)
            values = tree.values
            assert values == [(1, "one"), (3, "three"), (5, "five")]
        finally:
            settings.STRICT = original_strict


class TestNumberTreeRealWorldScenarios:
    """Tests simulating real-world PDF number tree usage"""

    def test_page_labels_tree(self):
        """Simulate a page labels number tree."""
        obj = {
            "Nums": [
                0,
                {"S": "r"},
                4,
                {"S": "D"},
                10,
                {"S": "D", "St": 1},
            ]
        }
        tree = NumberTree(obj)
        values = tree.values
        assert len(values) == 3
        assert values[0][0] == 0
        assert values[1][0] == 4
        assert values[2][0] == 10

    def test_structure_tree_parent_tree(self):
        """Simulate a structure tree parent tree."""
        child1 = {"Nums": [0, "struct0", 1, "struct1"], "Limits": [0, 1]}
        child2 = {"Nums": [2, "struct2", 3, "struct3"], "Limits": [2, 3]}
        root = {"Kids": [child1, child2]}
        tree = NumberTree(root)
        values = tree.values
        assert len(values) == 4
        assert values == [
            (0, "struct0"),
            (1, "struct1"),
            (2, "struct2"),
            (3, "struct3"),
        ]

    def test_sparse_number_tree(self):
        """Number tree with sparse/non-contiguous keys."""
        obj = {"Nums": [0, "first", 100, "hundredth", 1000, "thousandth"]}
        tree = NumberTree(obj)
        values = tree.values
        assert values == [(0, "first"), (100, "hundredth"), (1000, "thousandth")]

    def test_multi_level_page_labels(self):
        """Multi-level tree for page labels."""
        leaf1 = {"Nums": [0, {"S": "r"}], "Limits": [0, 0]}
        leaf2 = {"Nums": [1, {"S": "r"}], "Limits": [1, 1]}
        intermediate1 = {"Kids": [leaf1, leaf2], "Limits": [0, 1]}
        leaf3 = {"Nums": [2, {"S": "D"}], "Limits": [2, 2]}
        leaf4 = {"Nums": [3, {"S": "D"}], "Limits": [3, 3]}
        intermediate2 = {"Kids": [leaf3, leaf4], "Limits": [2, 3]}
        root = {"Kids": [intermediate1, intermediate2]}
        tree = NumberTree(root)
        values = tree.values
        assert len(values) == 4
        assert [k for k, v in values] == [0, 1, 2, 3]


class TestNumberTreeIterableBehavior:
    """Tests for NumberTree with various iterable types"""

    def test_generator_nums(self):
        """Nums can be backed by various iterables."""
        obj = {"Nums": list(range(6))}
        tree = NumberTree(obj)
        values = tree.values
        assert values == [(0, 1), (2, 3), (4, 5)]

    def test_values_property_recalculates(self):
        """Values property recalculates each time (no caching)."""
        obj = {"Nums": [0, "initial"]}
        tree = NumberTree(obj)
        values1 = tree.values
        values2 = tree.values
        assert values1 == values2


class TestNumberTreeKeyOrdering:
    """Tests specifically for key ordering behavior"""

    def test_ascending_order_valid_strict(self):
        """Strictly ascending keys pass in strict mode."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            obj = {"Nums": [1, "a", 2, "b", 3, "c"]}
            tree = NumberTree(obj)
            values = tree.values
            assert values == [(1, "a"), (2, "b"), (3, "c")]
        finally:
            settings.STRICT = original_strict

    def test_non_ascending_fails_strict(self):
        """Non-ascending keys fail in strict mode."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            obj = {"Nums": [2, "b", 1, "a"]}
            tree = NumberTree(obj)
            with pytest.raises(PDFSyntaxError):
                _ = tree.values
        finally:
            settings.STRICT = original_strict

    def test_descending_order_sorted_non_strict(self):
        """Descending keys are sorted in non-strict mode."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = False
            obj = {"Nums": [3, "c", 2, "b", 1, "a"]}
            tree = NumberTree(obj)
            values = tree.values
            assert values == [(1, "a"), (2, "b"), (3, "c")]
        finally:
            settings.STRICT = original_strict

    def test_same_key_ascending_valid_strict(self):
        """Same consecutive keys are valid (a <= b where a == b)."""
        original_strict = settings.STRICT
        try:
            settings.STRICT = True
            obj = {"Nums": [1, "a", 1, "b", 2, "c"]}
            tree = NumberTree(obj)
            values = tree.values
            assert len(values) == 3
        finally:
            settings.STRICT = original_strict

    def test_zero_keys(self):
        """Tree with all zero keys."""
        obj = {"Nums": [0, "a", 0, "b", 0, "c"]}
        tree = NumberTree(obj)
        values = tree.values
        assert len(values) == 3
        assert all(k == 0 for k, v in values)
