.PHONY: setup api frontend dev lint test clean

setup:
	cd api && pip install -r requirements.txt
	cd frontend && npm install

api:
	cd api && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

dev:
	@echo "Starting API and Frontend in parallel..."
	$(MAKE) api & $(MAKE) frontend & wait

lint:
	cd api && ruff check .
	cd frontend && npm run lint

test:
	cd api && pytest -v

clean:
	docker compose down -v
	rm -rf frontend/.next frontend/node_modules
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
