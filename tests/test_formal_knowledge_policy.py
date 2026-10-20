"""Approved shipping identities must survive reading each policy in isolation."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name", ["product-faq.md", "billing-shipping.md", "member-benefits.md"]
)
def test_formal_shipping_states_identity_price_and_full_points_offset(name):
    text = (ROOT / "knowledge" / name).read_text().replace(" ", "")
    assert "普通档用户未满99元" in text and "10元基础运费" in text
    assert "银卡、金卡用户未满99元" in text and "6元基础运费" in text
    assert "600积分可全额抵扣这6元基础运费" in text
    assert "还需支付4元" not in text
    assert "偏远地区" in text


def test_existing_other_membership_rights_remain_explicit():
    text = (ROOT / "knowledge/member-benefits.md").read_text()
    for clause in [
        "银卡享商品 9 折",
        "金卡享商品 95 折",
        "金卡会员每月享 3 次免运费",
        "银卡会员每月 1 次",
        "单笔最多抵 30 元",
        "积分有效期 24 个月",
    ]:
        assert clause in text
