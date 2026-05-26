def run() -> None:
    from .bot import Bot
    from .config import Config
    from .logging_config import setup_logging
    from .store import Store

    config = Config.load()
    setup_logging(config.log_dir)

    store = Store(data_dir=config.data_dir, log_dir=config.log_dir, input_budget=config.token_budget)
    bot = Bot(store=store, config=config)
    bot.start()

if __name__ == "__main__":
    run()
