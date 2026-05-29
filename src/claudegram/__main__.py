def run() -> None:
    from .bot import Bot
    from .config import Config
    from .credentials import CredentialStore
    from .logging_config import setup_logging
    from .secrets import Secrets
    from .store import Store

    config = Config.load()
    setup_logging(config.log_dir)

    store = Store(data_dir=config.data_dir, log_dir=config.log_dir, input_budget=config.token_budget)
    secrets = Secrets(config.credential_enc_key)
    credentials = CredentialStore(
        data_dir=config.data_dir, secrets=secrets, pool_api_key=config.claude_api_key,
    )
    bot = Bot(store=store, config=config, credentials=credentials)
    bot.start()

if __name__ == "__main__":
    run()
