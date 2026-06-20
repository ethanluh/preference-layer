"""Walk the PTP HTTP surface end-to-end, including the RFC 8628 device flow (A2).

This drives the real FastAPI app in-process (no port needed) so it doubles as a
runnable smoke check for the device-flow wiring:

    agent  POST /device/code      -> device_code + user_code
    agent  POST /token            -> 400 authorization_pending
    owner  GET  /device           -> sees the requested scope
    owner  POST /device/decision  -> approve
    agent  POST /token            -> Bearer token
    agent  GET  /preference       -> signed, scoped credential

    pip install 'preferencelayer[dev]'   # FastAPI + httpx
    python experiments/run_ptp_api.py            # in-process walkthrough
    python experiments/run_ptp_api.py --serve    # bind a real port instead
"""

from __future__ import annotations

import argparse

from preferencelayer.http import build_app
from preferencelayer.http.app import DEVICE_CODE_GRANT
from preferencelayer.ptp.credential import (
    AttributeNode,
    PreferenceCredential,
    PreferenceGraph,
    new_user_keypair,
)
from preferencelayer.ptp.device_flow import DeviceFlowAuthority
from preferencelayer.ptp.store import CredentialStore


def build_demo_store() -> CredentialStore:
    sk, did = new_user_keypair(seed=b"ptp-demo-seed-000000000000000000"[:32])
    store = CredentialStore(sk, did)
    store.put_credential(PreferenceCredential(did, PreferenceGraph(
        category="laptops",
        attributeNodes=[
            AttributeNode("performance", 0.8, 0.7),
            AttributeNode("portability", 0.6, 0.5),
            AttributeNode("price_sensitivity", -0.3, 0.6),
        ],
    )))
    return store


def walkthrough() -> None:
    from fastapi.testclient import TestClient

    store = build_demo_store()
    # interval=0 so the walkthrough can poll back-to-back.
    authority = DeviceFlowAuthority(store, interval=0)
    client = TestClient(build_app(store, device_authority=authority))

    code = client.post("/device/code", json={"client_id": "agent.shop", "scope": ["laptops"]}).json()
    print(f"POST /device/code      -> user_code={code['user_code']} verify={code['verification_uri_complete']}")

    pending = client.post("/token", json={"grant_type": DEVICE_CODE_GRANT, "device_code": code["device_code"]})
    print(f"POST /token (pending)  -> {pending.status_code} {pending.json()['detail']['error']}")

    seen = client.get("/device", params={"user_code": code["user_code"]}).json()
    print(f"GET  /device           -> owner sees scope={seen['scope']}")

    client.post("/device/decision", json={"user_code": code["user_code"], "decision": "approve"})
    print("POST /device/decision  -> approved")

    granted = client.post("/token", json={"grant_type": DEVICE_CODE_GRANT, "device_code": code["device_code"]}).json()
    token = granted["access_token"]
    print(f"POST /token (approved) -> {granted['token_type']} token, expires_in={granted['expires_in']}s")

    headers = {"Authorization": f"Bearer {token}"}
    pref = client.request("GET", "/preference", json={"category": "laptops"}, headers=headers)
    body = pref.json()
    cred = PreferenceCredential.from_dict(body["credential"])
    print(f"GET  /preference       -> {pref.status_code} coverage={body['coverage']} "
          f"confidence={body['confidence']} signature_valid={cred.verify(store.signing_key.verify_key)}")

    other = client.request("GET", "/preference", json={"category": "headphones"}, headers=headers)
    print(f"GET  /preference (oos) -> {other.status_code} (token scoped to laptops only)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="bind a real HTTP port instead of the in-process walkthrough")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if args.serve:
        import uvicorn

        uvicorn.run(build_app(build_demo_store()), host=args.host, port=args.port)
    else:
        walkthrough()


if __name__ == "__main__":
    main()
