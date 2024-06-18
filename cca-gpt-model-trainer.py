#!/usr/bin/env python

import os
from art import text2art
from colorama import init, Fore, Style

import MySQLdb
import logging
import re
from bs4 import BeautifulSoup
import requests
from requests.auth import HTTPBasicAuth
import json
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from torch.optim import AdamW
from datasets import load_dataset, DatasetDict
from torch.cuda import OutOfMemoryError
import torch
import torch.nn as nn


# Init Logging
logging.basicConfig(level=logging.ERROR)

# Initialize colorama
init(autoreset=True)

# Clear the screen
os.system('cls' if os.name == 'nt' else 'clear')

# Print a fancy title using ANSI escape codes
# Generate the title using the art module
title_art = text2art("- cca-gpt-model-trainer -", font="tarty4")
indent = ":: "

# Print the title with color and shadow effect
print(Fore.GREEN + Style.BRIGHT + title_art)
print(Fore.CYAN + Style.BRIGHT + "\nVersion 1.0 // Written by Dean Thomson, for CCA Software (c) 2024\n")

# Ensure the output directory exists
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

def fetch_mediawiki_data():

    # Fetches data from MediaWiki and saves it to a text file.
    # The text file is named 'mediawiki_content.txt' in the 'output' directory.
    
    try:
        task = "Fetching MediaWiki SQL Data."
        print(Fore.YELLOW + indent + Fore.LIGHTCYAN_EX + task)
        connection = MySQLdb.connect(
            host="localhost",
            user="your_mysql_user",        # Change this setting
            password="your_mysql_password",# Change this setting
            database="your_database"       # Change this setting
        )
        cursor = connection.cursor()
        query = """
            SELECT page_title, old_text
            FROM page
            JOIN revision ON page.page_id = revision.rev_page
            JOIN text ON revision.rev_id = text.old_id;
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        output_file = os.path.join(output_dir, 'mediawiki_content.txt')
        with open(output_file, 'w', encoding='utf-8') as file:
            for row in rows:
                file.write(f"{row[0]}\n{row[1]}\n")

        print(Fore.GREEN + indent + Fore.LIGHTCYAN_EX + "MediaWiki data fetched and saved.")
    except Exception as e:
        logging.error(f"Error fetching MediaWiki data: {e}")
        print(Fore.RED + f"Error fetching MediaWiki data: {e}")

def clean_mediawiki_data():

    # Cleans MediaWiki data by removing HTML tags.
    # The cleaned data is saved to a new text file.
    
    try:
        task = "Cleaning MediaWiki Data."
        print(Fore.YELLOW + indent + Fore.LIGHTCYAN_EX + task)
        
        input_file = os.path.join(output_dir, 'mediawiki_content.txt')
        output_file = os.path.join(output_dir, 'cleaned_mediawiki_content.txt')
        
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()
        
        cleaned_content = re.sub(r'<[^>]+>', '', content)
        
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(cleaned_content)
        
        print(Fore.GREEN + indent + Fore.LIGHTCYAN_EX + "MediaWiki data cleaned.")
    except Exception as e:
        logging.error(f"Error cleaning MediaWiki data: {e}")
        print(Fore.RED + f"Error cleaning MediaWiki data: {e}")

def fetch_jira_data():

    # Fetches Jira data and saves it to a JSON file.
    # Jira data is fetched from a Jira Cloud instance.
    # The data is saved to a JSON file.
    
    try:
        task = "Fetching Jira Entries from Jira Cloud."
        print(Fore.YELLOW + indent + Fore.LIGHTCYAN_EX + task)
        url = 'https://your_jira_instance.atlassian.net/rest/api/3/search'  # Change this setting
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        params = {
            'jql': 'project = YOUR_PROJECT_KEY',  # Change this setting
            'maxResults': 3000,
            'fields': 'summary,description,comment'
        }

        email = 'your_email'                 # Change this setting
        api_token = 'your_api_token'         # Change this setting
        
        response = requests.get(url, headers=headers, params=params, auth=HTTPBasicAuth(email, api_token))

        if response.status_code == 200:
            jira_data = response.json()
            output_file = os.path.join(output_dir, 'jira_issues.json')
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(jira_data, file, indent=4)
            
            print(Fore.GREEN + indent + Fore.LIGHTCYAN_EX + "Jira data fetched and saved.")
        else:
            logging.error(f"Error fetching Jira data: HTTP {response.status_code}")
            print(Fore.RED + f"Error fetching Jira data: HTTP {response.status_code}")
    except Exception as e:
        logging.error(f"Error fetching Jira data: {e}")
        print(Fore.RED + f"Error fetching Jira data: {e}")

def combine_files(jira_file, mediawiki_file, combined_file):
    
    # Combines Jira and MediaWiki data into a single text file for training.
    # Jira data is fetched from a JSON file, while MediaWiki data is fetched from a text file.
    # The combined data is written to a new text file.
    
    try:
        task = "Combining Jira and MediaWiki Data."
        print(Fore.YELLOW + indent + Fore.LIGHTCYAN_EX + task)
        
        with open(jira_file, 'r', encoding='utf-8') as file:
            jira_data = json.load(file)
        
        jira_content = ""
        for issue in jira_data['issues']:
            summary = issue['fields']['summary']
            description = issue['fields'].get('description', '')
            jira_content += f"{summary}\n{description}\n"
        
        with open(mediawiki_file, 'r', encoding='utf-8') as file:
            mediawiki_content = file.read()
        
        with open(combined_file, 'w', encoding='utf-8') as file:
            file.write(jira_content)
            file.write(mediawiki_content)
        
        print(Fore.GREEN + indent + Fore.LIGHTCYAN_EX + "Data combined for training.")
    except Exception as e:
        logging.error(f"Error combining data: {e}")
        print(Fore.RED + f"Error combining data: {e}")

def download_tokenizer_files(model_name, output_dir):
    
    # Downloads the tokenizer files for the specified model.
    # Args: model_name (str): The name of the model to download the tokenizer files for.
    #       output_dir (str): The directory where the tokenizer files will be saved.
    
    try:
        task = "Downloading Tokenizer Files."
        print(Fore.YELLOW + indent + Fore.LIGHTCYAN_EX + task)
        
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
        tokenizer.save_pretrained(output_dir)
        
        print(Fore.GREEN + indent + Fore.LIGHTCYAN_EX + "Tokenizer files downloaded.")
    except Exception as e:
        logging.error(f"Error downloading tokenizer files: {e}")
        print(Fore.RED + f"Error downloading tokenizer files: {e}")

def save_model_and_tokenizer(model, tokenizer, output_dir):
    
    # Save model and tokenizer
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def fine_tune_gpt_model(data_file, output_dir):
    try:
        task = "Fine-Tuning GPT Model."
        print(task)
        
        model_name = "distilgpt2"
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
        
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

        # Load and split dataset into train and validation
        dataset = load_dataset('text', data_files=data_file)
        dataset = dataset['train'].train_test_split(test_size=0.1)
        
        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,  # Adjust the number of epochs as needed
            per_device_train_batch_size=1,  # Adjust batch size based on your GPU memory
            gradient_accumulation_steps=16,  # Accumulate gradients to simulate a larger batch size
            learning_rate=5e-5,  # Fine-tuned learning rate
            weight_decay=0.01,  # Apply weight decay for regularization
            warmup_steps=10,
            logging_dir='./logs',  # Enable logging directory
            logging_steps=10,  # Set logging steps to a positive value
            save_steps=500,
            save_total_limit=2,
            eval_strategy="epoch",  # Use `eval_strategy` instead of `evaluation_strategy`
            save_strategy="epoch",  # Ensure save strategy matches eval strategy
            load_best_model_at_end=True,  # Load the best model at the end of training
            report_to=None  # Disable reporting to any logging services like TensorBoard
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['test']
        )
        
        try:
            trainer.train()
            save_model_and_tokenizer(model, tokenizer, output_dir)
        except OutOfMemoryError as e:
            logging.error(f"GPU out of memory: {e}")
            print("GPU out of memory. Consider reducing batch size, using gradient accumulation, or switching to CPU.")
            # Retry on CPU
            print("Retrying on CPU...")
            training_args.no_cuda = True
            trainer.args = training_args
            trainer.train()
            save_model_and_tokenizer(model, tokenizer, output_dir)
        
    except Exception as e:
        logging.error(f"Error during model fine-tuning: {e}")
        print(f"Error during model fine-tuning: {e}")

def save_model_and_tokenizer(model, tokenizer, output_dir):
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def main():
    
    # Fetches data from MediaWiki, cleans it, fetches data from Jira, combines the data,
    # downloads tokenizer files, and fine-tunes a GPT model on the combined data.

    fetch_mediawiki_data()
    clean_mediawiki_data()
    fetch_jira_data()

    jira_file = os.path.join(output_dir, 'jira_issues.json')
    mediawiki_file = os.path.join(output_dir, 'cleaned_mediawiki_content.txt')
    combined_file = os.path.join(output_dir, 'combined_training_data.txt')
    combine_files(jira_file, mediawiki_file, combined_file)

    download_tokenizer_files("distilgpt2", output_dir)
    fine_tune_gpt_model(combined_file, output_dir)

if __name__ == '__main__':
    main()