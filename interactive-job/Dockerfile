FROM python:3.10-slim

WORKDIR /app

COPY run_interactive_assets.py /app/run_interactive_assets.py
COPY run_interactive.sh /app/run_interactive.sh

RUN chmod +x /app/run_interactive.sh

ENTRYPOINT ["/app/run_interactive.sh"]
