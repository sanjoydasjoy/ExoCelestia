# Exoplanet Detection API – Reference Guide

## Base URL

Local development backend: `http://localhost:8000`
(All routes are prefixed with `/api`)

---

## Authentication

The API uses **JWT Bearer tokens**.

Acquire a token via `POST /api/auth/login` or `POST /api/auth/signup`, then include the header:

```http
Authorization: Bearer <your-access-token>
```

---

## Endpoints

### 1. `POST /api/auth/signup` – Register

Registers a new user and returns an auth token.

```jsonc
Request
{
  "email": "astronaut@nasa.gov",
  "password": "SecurePass123",
  "full_name": "John Doe" // optional
}
```

```jsonc
Success ‑ 201
{
  "access_token": "<jwt>",
  "token_type": "bearer",
  "user": {
    "id": "uuid",
    "email": "astronaut@nasa.gov",
    "full_name": "John Doe",
    "created_at": "2025-10-10T12:00:00Z",
    "is_active": true
  }
}
```

---

### 2. `POST /api/auth/login` – Login

Same payload as signup (without `full_name`). Returns the same success body.

Errors: `401 Unauthorized – Incorrect email or password`

---

### 3. `GET /api/auth/me` – Current User _(protected)_

Returns the authenticated user object.

---

### 4. `POST /api/auth/verify` – Verify Token _(protected)_

Verifies a token and returns the user if valid.

---

### 5. `POST /api/predict` – Single Prediction _(protected)_

```jsonc
Request
{
  "features": {
    "orbital_period": 3.5,
    "transit_depth": 0.02,
    "transit_duration": 0.15,
    "stellar_radius": 1.2,
    "stellar_mass": 1.0
  }
}
```

```jsonc
Success ‑ 200
{
  "prediction": "confirmed",          // confirmed | candidate | false_positive
  "confidence": 0.87,
  "explain": {
    "top_features": [
      { "name": "orbital_period", "value": 0.80 },
      { "name": "transit_depth",  "value": 0.65 }
    ]
  }
}
```

Possible errors:
* `400 Bad Request – Model file not found / invalid features`
* `500 Internal Server Error – Prediction failed`

---

### 6. `POST /api/predict/batch` – CSV Batch Prediction _(protected)_

**Multipart form-data** field `file` must be a `.csv`.

```csv
orbital_period,transit_depth,transit_duration,stellar_radius,stellar_mass
3.5,0.02,0.15,1.2,1.0
5.2,0.01,0.18,0.9,0.8
```

```jsonc
Success ‑ 200
{
  "message": "Successfully processed 10 rows (2 rows had errors)",
  "rows_processed": 10,
  "predictions": [
    {
      "row_index": 0,
      "prediction": "confirmed",
      "confidence": 0.87,
      "explain": { "top_features": [ ... ] }
    }
  ]
}
```

Possible errors:
* `400 – Only CSV files are accepted`
* `400 – No valid predictions made`
* `500 – CSV processing failed`

---

### 7. `GET /api/model/info` – Model Metadata _(protected)_

Returns
```jsonc
{
  "model_path": "ml/models/model.pkl",
  "model_type": "SklearnModelLoader",
  "feature_names": ["orbital_period", "transit_depth", ...],
  "config": { /* model configuration */ }
}
```

---

## Error Format

```json
{
  "detail": "Human-readable error message"
}
```

---

## Status Codes
| Code | Meaning |
|------|---------|
|200|Success / OK|
|201|Created (signup)|
|400|Bad request / validation error|
|401|Unauthorized (missing/invalid token)|
|500|Internal server error|

---

## Quick cURL Example

```bash
# Sign up and save the token
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@nasa.gov","password":"SecurePass123"}' | jq -r .access_token)

# Single prediction
curl -X POST http://localhost:8000/api/predict \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"features":{"orbital_period":3.5,"transit_depth":0.02,"transit_duration":0.15,"stellar_radius":1.2,"stellar_mass":1.0}}'
```

---

## Security Notes
* Passwords stored hashed with **bcrypt**.
* Tokens signed with **HS256**; set `SECRET_KEY` via env in production.
* CORS is `*` in dev – tighten for prod.

---

_Last updated: 2025-10-10_
