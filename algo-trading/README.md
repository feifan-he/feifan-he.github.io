
# Algorithmic Trading


## System Requirements and Design
- Security - avoid storing password directly on the virtual machine, or sending password through Telegram app
	- Password cryptography technique will be used to securely authenticate to Robinbood
		- Encrypted password will be stored on the Google's VM
		- Encryption key will be sent via Telegram App
		- Password will be decrypted during the initialization phase and it will be used to authenticate to Robinhood
- Execution - how to run the trade algorithm
	- The algorithm will retrieve historical data every half an hour
	- Trade algorithms will be run every half an hour. It will look at stock prices, and buy/sell signals will be generated during this time
	- Need a manual override to trigger buy/sell order
	- Weight mechanism - able to change the % of the total equity to invest in
- Investment Status - able to track balances, and daily changes
	- Need to be able to track daily profit/loss for each stock
	- Need a generated chart to view weekly changes

## Tech Stack
- Python3
- Pip3
- [robinhood-unofficial/pyrh](https://github.com/robinhood-unofficial/pyrh)
- [python-telegram-bot/python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)

## Execution Flow
### Initialization
```mermaid
sequenceDiagram
participant TA as Telegram App
participant GCS as Google Cloud Server
participant R as Robinhood API
TA ->> GCS: Start/Reboot Application
GCS ->> TA: What's the key?
TA ->> GCS: Send key
GCS ->> GCS: Use key to decrypt credentials(For Security)
GCS ->> R: Authenticate using decrypted credentials
GCS ->> TA: If failed, ask user to resend key
TA ->> GCS: Set algorithm configs
TA ->> GCS: Start running trade algorithm
GCS ->> GCS: Run trade algorithm
```

### On Price Update (Every Half an Hour)
```mermaid
sequenceDiagram
participant TA as Telegram App
participant GCS as Google Cloud Server
participant R as Robinhood API
GCS ->> R: Run hourly: Retrieve stock prices
GCS ->> GCS: Compute trade signals
GCS ->> R: Send buy/sell order if applicable
GCS ->> GCS: Retry if failed
GCS ->> TA: Notify order status
```

### Profit/Loss (Daily)
```mermaid
sequenceDiagram
participant TA as Telegram App
participant GCS as Google Cloud Server
participant R as Robinhood API
GCS ->> R: Retrieve balance
GCS ->> GCS: Computer profit/loss
GCS ->> TA: Send daily profit/loss
```

## In Action
![](http://feifanhe.com/algo-trading/img6.png)
