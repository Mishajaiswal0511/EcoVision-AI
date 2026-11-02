"""
test_polygon_conversion.py  –  EcoVision AI
===========================================
Unit tests for polygon-to-bounding-box parsing, input validation,
boundary tolerances, and the 6-class target taxonomy.
"""

import json
import math
import pytest
from pathlib import Path

from scripts.label_mapper import parse_annotation_to_bbox, COORD_TOLERANCE


class TestPolygonToBBoxParsing:
    """Tests for parse_annotation_to_bbox logic."""

    def test_valid_quadrilateral_polygon(self):
        # 4 vertices: (0.1, 0.2), (0.5, 0.2), (0.5, 0.8), (0.1, 0.8)
        # Expected: x_min=0.1, x_max=0.5 -> cx=0.3, w=0.4
        #           y_min=0.2, y_max=0.8 -> cy=0.5, h=0.6
        tokens = ["0", "0.1", "0.2", "0.5", "0.2", "0.5", "0.8", "0.1", "0.8"]
        result = parse_annotation_to_bbox(tokens)
        assert result is not None
        cls_id, cx, cy, w, h = result
        assert cls_id == 0
        assert pytest.approx(cx, abs=1e-5) == 0.3
        assert pytest.approx(cy, abs=1e-5) == 0.5
        assert pytest.approx(w, abs=1e-5) == 0.4
        assert pytest.approx(h, abs=1e-5) == 0.6

    def test_minimal_triangle_polygon(self):
        # 3 vertices (6 coords): (0.2, 0.3), (0.4, 0.7), (0.6, 0.5)
        # x_min=0.2, x_max=0.6 -> cx=0.4, w=0.4
        # y_min=0.3, y_max=0.7 -> cy=0.5, h=0.4
        tokens = ["2", "0.2", "0.3", "0.4", "0.7", "0.6", "0.5"]
        result = parse_annotation_to_bbox(tokens)
        assert result is not None
        cls_id, cx, cy, w, h = result
        assert cls_id == 2
        assert pytest.approx(cx, abs=1e-5) == 0.4
        assert pytest.approx(cy, abs=1e-5) == 0.5
        assert pytest.approx(w, abs=1e-5) == 0.4
        assert pytest.approx(h, abs=1e-5) == 0.4

    def test_odd_coordinate_count_rejected(self):
        # 5 coordinates (odd count, invalid polygon)
        tokens = ["0", "0.1", "0.2", "0.3", "0.4", "0.5"]
        assert parse_annotation_to_bbox(tokens) is None

        # 7 coordinates (odd count)
        tokens = ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"]
        assert parse_annotation_to_bbox(tokens) is None

    def test_fewer_than_3_points_rejected_as_polygon(self):
        # 2 coordinates (1 point)
        tokens = ["0", "0.1", "0.2"]
        assert parse_annotation_to_bbox(tokens) is None

    def test_non_finite_coordinates_rejected(self):
        tokens_nan = ["0", "nan", "0.2", "0.5", "0.2", "0.5", "0.8"]
        assert parse_annotation_to_bbox(tokens_nan) is None

        tokens_inf = ["0", "0.1", "inf", "0.5", "0.2", "0.5", "0.8"]
        assert parse_annotation_to_bbox(tokens_inf) is None

    def test_out_of_bounds_coordinates_rejected(self):
        # Coordinate exceeds 1.0 by more than tolerance
        tokens_high = ["0", "0.1", "0.2", "1.05", "0.2", "0.5", "0.8"]
        assert parse_annotation_to_bbox(tokens_high) is None

        # Coordinate is negative beyond tolerance
        tokens_low = ["0", "-0.05", "0.2", "0.5", "0.2", "0.5", "0.8"]
        assert parse_annotation_to_bbox(tokens_low) is None

    def test_floating_point_tolerance_clamping(self):
        # Coordinate slightly over 1.0 within tolerance (1e-4)
        tokens = ["1", "0.1", "0.1", "1.00005", "0.1", "0.5", "0.8"]
        result = parse_annotation_to_bbox(tokens, coord_tolerance=1e-4)
        assert result is not None
        cls_id, cx, cy, w, h = result
        assert cls_id == 1
        # Should be clamped to 1.0
        assert cx + w / 2.0 <= 1.000001
        assert cy + h / 2.0 <= 1.000001

    def test_degenerate_zero_area_polygon_rejected(self):
        # Vertical line: all xs are 0.3 -> w = 0
        tokens_line = ["0", "0.3", "0.1", "0.3", "0.5", "0.3", "0.9"]
        assert parse_annotation_to_bbox(tokens_line) is None

        # Single point repeated 3 times -> w = 0, h = 0
        tokens_point = ["0", "0.4", "0.4", "0.4", "0.4", "0.4", "0.4"]
        assert parse_annotation_to_bbox(tokens_point) is None

    def test_standard_5_token_bbox_passthrough(self):
        # Standard YOLO bbox: class cx cy w h
        tokens = ["3", "0.5", "0.5", "0.2", "0.3"]
        result = parse_annotation_to_bbox(tokens)
        assert result is not None
        cls_id, cx, cy, w, h = result
        assert cls_id == 3
        assert (cx, cy, w, h) == (0.5, 0.5, 0.2, 0.3)

    def test_standard_bbox_out_of_bounds_rejected(self):
        # cx = 0.9, w = 0.4 -> x2 = 1.1 (exceeds canvas)
        tokens = ["0", "0.9", "0.5", "0.4", "0.2"]
        assert parse_annotation_to_bbox(tokens) is None

    def test_negative_class_id_rejected(self):
        tokens = ["-1", "0.5", "0.5", "0.2", "0.2"]
        assert parse_annotation_to_bbox(tokens) is None

    def test_non_integer_class_id_rejected(self):
        tokens = ["cat", "0.5", "0.5", "0.2", "0.2"]
        assert parse_annotation_to_bbox(tokens) is None


class TestSixClassMappingPreservation:
    """Verifies that the target taxonomy and mapping integrity remain unchanged."""

    def test_target_classes_exact(self):
        mapping_path = Path("data/label_mapping.json")
        with open(mapping_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        expected = ["plastic", "metal", "cardboard", "glass", "paper", "general_trash"]
        assert cfg["target_classes"] == expected
        assert len(cfg["target_classes"]) == 6

    def test_excluded_classes_exact(self):
        mapping_path = Path("data/label_mapping.json")
        with open(mapping_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        mappings = cfg["mappings"]
        excluded = [k for k, v in mappings.items() if v is None]
        assert sorted(excluded) == ["Battery", "Carded blister pack", "Food waste", "Shoe"]

    def test_all_mappings_resolve_to_valid_targets(self):
        mapping_path = Path("data/label_mapping.json")
        with open(mapping_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        targets = set(cfg["target_classes"])
        for src, tgt in cfg["mappings"].items():
            if tgt is not None:
                assert tgt in targets, f"Source '{src}' mapped to invalid target '{tgt}'"

