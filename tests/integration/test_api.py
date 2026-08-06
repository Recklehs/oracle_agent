from fastapi.testclient import TestClient

from oracle_agent.app import app


class Test헬스체크:
    def test_헬스체크를_호출하면_정상_상태를_반환한다(self):
        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
