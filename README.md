# Smart Contract Wallet Generator App

This React app allows a user to generate a smart contract wallet using account abstraction--and then use it to send and/or receive MATIC. Smart contract wallet accounts provide advantages over traditional externally owned wallet accounts (EOA). For example, they are more secure and programmable, which gives users greater control over their assets. 

## Description

This app uses two primary Solidity contracts: SimpleAccountFactory.sol, and SimpleAccount.sol; they are both in the account abstraction implementation contracts mentioned in EIP-4337. After downloading  and compiling these contracts (along with some additional interface and import files) from Eth-Infinitism, SimpleAccountFactory.sol was deployed to the Polygon Mumbai testnet using Hardhat. 

To view the repository containing the account abstraction implementation contracts mentioned in EIP-4337, go to the following website: https://github.com/eth-infinitism/account-abstraction/tree/develop/contracts/samples

To view the SimpleAccountFactory contract deployed on Mumbai, go to the following website: https://mumbai.polygonscan.com/address/0x631de80Af334568eb2b372096b38e2F024cAB8Cc

The mechanism used by this app is the factory method: if a user wants to create a new smart contract wallet account, they must call a function in the SimpleAccountFactory contract, which then deploys a brand new smart contract wallet account on the Mumbai testnet. A user may create as many new accounts as they wish. 

## Getting Started

### Creating a new smart contract wallet account

To use this app, it is required to have MetaMask installed (with the Polygon Mumbai testnet network added to it). To get some testnet MATIC, go to the following faucet: https://faucet.polygon.technology. 

To get started using the app to create smart contract wallet accounts, go to the following website: https://abstract-wallet-app.netlify.app.

Once you are on the app's website, connect to MetaMask by clicking on the "Connect MetaMask" button. Your account address should appear in red text under the button. Next, to create a new smart contract wallet account, click on the "Generate" button in the "Generate a New Wallet" section. MetaMask will then pop up for you to approve this transaction (you will need some Mumbai testnet MATIC to do this). After deployment to the Mumbai network, your new smart contract wallet account address will appear in red text under the "Generate" button. 

Copy and save your new wallet's address. 

### Using a smart contract wallet account 

To use a smart contract wallet account, first connect to it by entering its address and clicking on the "Get Balance" button. The account's current balance will be displayed under the button. 

To deposit funds, which will be necessary if it is a new account, enter the desired amount in the input box in the "Deposit Funds" section and click on the "Deposit" button. MetaMask will then pop up for you to approve this transaction.

To send funds, enter the desired amount and the recipient address in the appropriate input boxes in the "Send Funds" section and click on the "Send" button. MetaMask will then pop up for you to approve this transaction.
