<p align="center">
  <img src="readme_header.png" />
</p>

# CCA-GPT Model Trainer

This repository contains a script to automate the process of fetching, cleaning, and combining data from MediaWiki and Jira, and then fine-tuning a language model (GPT) on the combined dataset. The script is designed to handle GPU memory constraints and can switch to CPU if needed.

## Features

- **Data Extraction**: Fetches data from a local MediaWiki SQL database and Jira using the Jira REST API.
- **Data Cleaning**: Cleans MediaWiki formatting from the extracted data.
- **Data Combination**: Combines the cleaned MediaWiki data with Jira entries.
- **Model Training**: Fine-tunes a GPT model (using `distilgpt2` by default) on the combined dataset.
- **Error Handling**: Includes robust error handling for database connections, API requests, file operations, and GPU memory issues.
- **GPU Optimization**: Optimized for training on an Nvidia GPU with fallback to CPU if necessary.

## Installation

**Clone the Repository**:
   ```sh
   git clone https://github.com/grahfmusic/cca-gpt-model-trainer.git
   cd cca-gpt-model-trainer
   ```

**Install Dependencies**:
   ```sh
   pip install MySQLdb beautifulsoup4 requests transformers datasets torch
   ```

**Configure Credentials**:
   - Update the database credentials and Jira credentials in the script (`cca-gpt-model-trainer.py`).

## Pre-requisites

**Export the Wiki Database from Host Machine**:
The Wiki database (`wiki`) uses MariaDB. To export the database, run a mysqldump with the appropriate user and database.
An example:

     mysqldump -u root -p wiki > wiki_dump.sql

**Create a New Database and Import the Dump**:
To create a new database called `local_wiki` with the username `grahf` and import the dump into it, run the following commands:
     sudo mysql -u root -p
     CREATE DATABASE local_wiki;
     CREATE USER 'grahf'@'localhost' IDENTIFIED BY 'd3anth0ms0n';
     GRANT ALL PRIVILEGES ON local_wiki.* TO 'dean'@'localhost';
     FLUSH PRIVILEGES;
     exit

     sudo mysql -u grahf -p local_wiki < wiki_dump.sql

This allows the script to export the data from the wikipedia without it directly accessing the production sql database.

## Usage

Run the script to automate the entire process:
```sh
python cca-gpt-model-trainer.py
```

## Script Workflow

1. **Fetch MediaWiki Data**: Connects to the local SQL database, retrieves page titles and content, and saves them to a text file.
2. **Clean MediaWiki Data**: Removes MediaWiki formatting from the text file and saves the cleaned content.
3. **Fetch Jira Data**: Fetches Jira entries using the Jira REST API and saves them to a JSON file.
4. **Combine Data**: Combines the cleaned MediaWiki data with Jira entries into a single training file.
5. **Fine-tune GPT Model**: Fine-tunes the specified GPT model on the combined dataset, with error handling for GPU memory issues and optional retry on CPU.
