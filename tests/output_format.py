from datetime import datetime, timedelta

# def generate_predictions():
#     output = {
#         "request_id": "ABC123",
#         "preds": [{
#             "P10": "2022-12-27 08:26:49.219717",
#             "P50": "2022-12-27 09:26:49.219717",
#             "P90": "2022-12-27 10:26:49.219717",
#         }],
#     }
#     return output

def generate_predictions(quantiles):
    base_time = datetime(2022, 12, 27, 8, 26, 49, 219717)
    output = {
        "request_id": "ABC123",
        "preds": []
    }
    for quantile in quantiles:
        output["preds"].append({
            f"P{int(quantile * 100)}": (base_time + timedelta(hours=quantile * 3)).isoformat()
        })
    return output