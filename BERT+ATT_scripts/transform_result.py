import json

path = "r/1/output/test-results-4.json"
data = json.load(open(path))
result = {"setting": "closed", "predictions": []}
for each in data:
	result["predictions"].append({"h_id": each['entpair'][0], "t_id": each['entpair'][1], "relation": each['relation'], "score": each['score']})
with open("r/1/output/test-results-4-output.json", "w") as f:
	json.dump(result, f)
