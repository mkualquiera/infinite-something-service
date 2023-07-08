from gradio import networking
from main import app
import uvicorn
import secrets


if __name__ == "__main__":
    port = networking.get_first_available_port(networking.INITIAL_PORT_VALUE, networking.INITIAL_PORT_VALUE + networking.TRY_NUM_PORTS)
    host = "0.0.0.0"
    config = uvicorn.Config(
                    app=app,
                    port=port,
                    host=host,
                    log_level="warning",
                    ws_max_size=1024 * 1024 * 1024,  # Setting max websocket size to be 1 GB
                )
    server = networking.Server(config=config)
    server.run_in_thread()
    print(f"Locally running on https://{host}:{port}")
    share_token = secrets.token_urlsafe(32)
    share_link = networking.setup_tunnel(host, port, share_token)
    print(f"Shared at {share_link}")
    server.thread.join()