import pytest
from datetime import datetime

# Import the function to be tested
from output_format import generate_predictions

@pytest.mark.parametrize("quantiles", [
    [0.1, 0.5, 0.9],
    # [0.1, 0.5, 0.9, 0.97]
    # Add more test cases as needed
])

def test_predictions_order_and_difference(quantiles):
    output = generate_predictions(quantiles)
    
    preds = [datetime.fromisoformat(pred[f"P{int(q * 100)}"]) for pred, q in zip(output["preds"], quantiles)]
    
    for i in range(len(preds) - 1):
        assert preds[i] <= preds[i + 1], f"P{int(quantiles[i] * 100)} should not be later than P{int(quantiles[i + 1] * 100)}"
        assert (preds[i + 1] - preds[i]).days < 3, f"Difference between P{int(quantiles[i] * 100)} and P{int(quantiles[i + 1] * 100)} should be less than 3 days"


@pytest.mark.skip(reason="Not implemented yet")
def test_input_format():
    pass




def test_predictions_order(quantiles):
    output = generate_predictions(quantiles)

    preds = output["preds"][0]

    p10 = datetime.fromisoformat(preds["P10"])
    p50 = datetime.fromisoformat(preds["P50"])
    p90 = datetime.fromisoformat(preds["P90"])

    assert p10 <= p50, "P10 should not be later than P50"
    assert p50 <= p90, "P50 should not be later than P90"


def test_predictions_date_difference(quantiles):
    output = generate_predictions(quantiles)

    preds = output["preds"][0]
    p10 = datetime.fromisoformat(preds["P10"])
    p50 = datetime.fromisoformat(preds["P50"])
    p90 = datetime.fromisoformat(preds["P90"])

    assert (p50 - p10).days < 3, "Difference between P10 and P50 should be less than 3 days"
    assert (p90 - p50).days < 3, "Difference between P50 and P90 should be less than 3 days"