import pytest

from wsd_torch_models import data_utils


@pytest.mark.parametrize("tags_to_filter_out", [None, set(["Z99"])])
def test_load_usas_mapper(tags_to_filter_out: set[str] | None) -> None:
    usas_mapper = data_utils.load_usas_mapper(tags_to_filter_out)
    assert isinstance(usas_mapper, dict)
    assert len(usas_mapper) > 0

    if tags_to_filter_out is None:
        assert len(usas_mapper) == 222
        assert "Z99" in usas_mapper
    else:
        assert len(usas_mapper) == 221
        assert "Z99" not in usas_mapper
    
    assert "A1.1.1" in usas_mapper
    expected_title_description = (
        "title: General actions, making etc. description: "
        "General/abstract terms relating to an activity/action "
        "(e.g. act, adventure, approach, arise); a characteristic/feature "
        "(e.g. absorb, attacking, automatically); "
        "aconstruction/craft and/or the action of constructing/crafting "
        "(e.g. arrange, assemble, bolts, boring, break)"
    )
    assert expected_title_description == usas_mapper["A1.1.1"]

    assert "A.1" not in usas_mapper
