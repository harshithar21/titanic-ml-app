GOOD = {"pclass": 1, "sex": "female", "age": 29, "sibsp": 0, "parch": 0, "fare": 80, "embarked": "S", "has_cabin": True}
BAD = {"pclass": 3, "sex": "male", "age": 30, "sibsp": 0, "parch": 0, "fare": 7.25, "embarked": "S", "has_cabin": False}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_loaded": True}


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "Titanic Survival Predictor" in r.text


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_name"] and 0.5 < body["test_metrics"]["accuracy"] <= 1


def test_predict_shape(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    b = r.json()
    assert set(b) == {"survived", "survival_probability", "prediction", "confidence", "model"}
    assert 0 <= b["survival_probability"] <= 1
    assert b["survived"] == (b["survival_probability"] >= 0.5)


def test_model_learned_sensible_pattern(client):
    good = client.post("/predict", json=GOOD).json()["survival_probability"]
    bad = client.post("/predict", json=BAD).json()["survival_probability"]
    assert good > 0.8 and bad < 0.2 and good > bad


def test_missing_age_is_allowed(client):
    r = client.post("/predict", json={**GOOD, "age": None})
    assert r.status_code == 200


def test_case_insensitive_inputs(client):
    r = client.post("/predict", json={**GOOD, "sex": "FEMALE", "embarked": "s"})
    assert r.status_code == 200


def test_invalid_inputs_rejected(client):
    for bad in ({**GOOD, "pclass": 5}, {**GOOD, "sex": "robot"}, {**GOOD, "age": -3}, {**GOOD, "fare": -1},
                {**GOOD, "embarked": "X"}, {k: v for k, v in GOOD.items() if k != "fare"}):
        assert client.post("/predict", json=bad).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"passengers": [GOOD, BAD]})
    assert r.status_code == 200 and r.json()["count"] == 2
    assert client.post("/predict/batch", json={"passengers": []}).status_code == 422
