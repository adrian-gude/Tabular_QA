from typing import List
import re
import ast

import pandas as pd
import torch
from databench_eval import utils
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Pipeline,
    pipeline,
)


class ColumnSelector:
    def __init__(self, pipe: Pipeline):
        self.pipe = pipe

    def select_relevant_columns(self, column_names: List[str], question: str):
        messages = [
            {
                "role": "system",
                "content": (
                    """
                    You are a tabular QA system specialized in understanding and analyzing datasets. Your task is to identify the most relevant columns from a given dataset that can answer a specific question.

                    You will be provided with a list of column names from the dataset.
                    Based on the question, analyze the provided column names and determine which ones are likely to contain the information required to answer the question. You only have to answer the question based on the provided column names in the formmating described below.
                
                    Input Format:
                        column_names: A list of column names from the dataset. Each column name is enclosed in single quotes and separated by commas. The column names may contain spaces and special characters.
                        question: A string containing the question to be answered.

                    Output Format:
                        A list of the relevant column names. The output should be a subset of the provided column names. Maintain the names EXACTLY as provided, special characters and all, for example < or >. If no columns are relevant, return an empty list. 
                        Only the relevant column names should be returned in list format, without any additional information or formatting.
                        
                    Example:
                        column_names: ['Name', 'Age', 'Email', 'Purchase Date', 'Product']
                        question: 'Which product was purchased?'
                        Output: ['Product']
                    
                    Input:
                        column_names: {column_names}
                        question: {question}

                    Output:
                    """
                ),
            },
            {
                "role": "user",
                "content": f"column_names: {column_names}\nquestion: {question}",
            },
        ]

        outputs = self.pipe(messages, max_new_tokens=2048, return_full_text=False)
        output = outputs[0]["generated_text"]
        return output

    # This function is only called for the selected columns, thus reducing the info passed to the model
    def extract_unique_column_values(self, df: pd.DataFrame, selected_columns):
        # selected_columns is given as a string, so we need to convert it to a list of strings. aLlow for the possibility of being denoted by either single or double quotes
        found_columns = re.findall(r"'(.*?)'", selected_columns)
        found_columns += re.findall(r'"(.*?)"', selected_columns)

        # Take the columns of the dataframe that are in the selected_columns list
        try:
            df = df[found_columns]
        except KeyError:
            return None

        # Remove all rows that contain any NaN values
        df = df.dropna()

        # Remove all columns that contain anything other than strings
        df = df.select_dtypes(include="category")

        # Count all the unique values of the columns
        counts = [df[col_name].nunique() for col_name in df.columns]

        # If the column has more than 20 unique values, remove it
        for i, col_name in enumerate(df.columns):
            if counts[i] > 20:
                df = df.drop(col_name, axis=1)

        # Print all the unique values of the columns
        result = [df[col_name].unique() for col_name in df.columns]

        result_str = ""

        for i, col_name in enumerate(df.columns):
            result_str += (
                f" # Column {col_name} can have the following values: {result[i]}\n"
            )

        if result_str != "":
            result_str = (
                "# The following columns contain a value from the following list :\n"
                + result_str
            )

        return result_str

    # This function is only called for the selected columns, thus reducing the info passed to the model
    def extract_dict_column_values(self, df: pd.DataFrame, selected_columns):
        # selected_columns is given as a string, so we need to convert it to a list of strings. aLlow for the possibility of being denoted by either single or double quotes
        found_columns = re.findall(r"'(.*?)'", selected_columns)
        found_columns += re.findall(r'"(.*?)"', selected_columns)

        # Take the columns of the dataframe that are in the selected_columns list
        try:
            df = df[found_columns]
        except KeyError:
            return None

        # Remove all rows that contain any NaN values
        df = df.dropna()

        # convert all columns to string
        df = df.astype(str)

        # Check for the columns that contain lists by searching if the string starts with a '[' and ends with a ']'. Trim the string to remove any leading or trailing whitespaces before checking
        for i, col_name in enumerate(df.columns):
            # print(df[col_name])
            try:
                if (
                    df[col_name][0].strip().startswith("{")
                    & df[col_name][0].strip().endswith("}")
                    is False
                ):
                    df = df.drop(col_name, axis=1)
            except (KeyError, IndexError):
                continue

        result_str = ""

        # For each column, we have to parse the string to a dictionary. The format of the dictionary is the following: 'key': value, 'key': value, ...
        for i, col_name in enumerate(df.columns):
            column = df[col_name]

            # Remove all rows that contain empty dicts
            column = column[column.str.strip() != "{}"]

            # Get the number of unique values in the column
            # unique_values = len(column.unique())
            rows = column.shape[0]

            keys = set()

            for j in range(rows):
                try:
                    try:
                        dict_str = ast.literal_eval(column.iloc[j])
                    except (ValueError, SyntaxError):
                        continue
                except KeyError:
                    continue

                try:
                    keys.update(dict_str.keys())
                except AttributeError:
                    continue

            # print(f"Column {col_name} has {rows} rows and {len(keys)} unique values")

            # If the number of unique values is greater than 25, then we skip the column
            if len(keys) > 25 or len(keys) == 0:
                continue

            result_str += f" # Column {col_name} is made of dictionaries that can contain the following keys: {keys}\n"

        if result_str != "":
            result_str = (
                "# The following columns are dictionaries that can contain the following keys :\n"
                + result_str
            )

        return result_str

        # The unique values of the columns used in the solution : {columns_unique}
        # The following columns are lists that can contain the following values: {columns_lists}
        # The following columns are dictionaries that can contain the following keys : {columns_dicts}

    # This function is only called for the selected columns, thus reducing the info passed to the model
    def extract_list_column_values(self, df: pd.DataFrame, selected_columns):
        # selected_columns is given as a string, so we need to convert it to a list of strings. aLlow for the possibility of being denoted by either single or double quotes
        found_columns = re.findall(r"'(.*?)'", selected_columns)
        found_columns += re.findall(r'"(.*?)"', selected_columns)

        # Take the columns of the dataframe that are in the selected_columns list
        try:
            df = df[found_columns]
        except KeyError:
            return None

        # Remove all rows that contain any NaN values
        df = df.dropna()

        # convert all columns to string
        df = df.astype(str)

        # Check for the columns that contain lists by searching if the string starts with a '[' and ends with a ']'. Trim the string to remove any leading or trailing whitespaces before checking
        for i, col_name in enumerate(df.columns):
            try:
                if (
                    df[col_name][0].strip().startswith("[")
                    & df[col_name][0].strip().endswith("]")
                    is False
                ):
                    df = df.drop(col_name, axis=1)
            except (KeyError, IndexError):
                continue

        result_str = "\n"

        # For each column, we have to parse the string to a list. The format of the list is the following: ['value', 'value', ...] or ["value", "value", ...] or [value, value, ...] and the separator can be either a comma or a semicolon
        for i, col_name in enumerate(df.columns):
            column = df[col_name]

            # Remove all rows that contain empty lists
            column = column[column.str.strip() != "[]"]

            # Get the number of unique values in the column
            # unique_values = len(column.unique())
            rows = column.shape[0]

            # result list
            values = []

            # Check for semicolons in the first 20 rows of the column. This is done to determine if the values are separated by commas or semicolons
            semicolon = False

            for j in range(rows):
                try:
                    if ";" in column.iloc[j]:
                        semicolon = True
                        break
                except KeyError:
                    continue

            # if there are semi-colons in the string, then we split by semi-colons
            if semicolon:
                for j in range(rows):
                    try:
                        values += (
                            df[col_name]
                            .iloc[j]
                            .strip()
                            .replace("[", "")
                            .replace("]", "")
                            .split(";")
                        )
                    except KeyError:
                        continue
            # Values are only separated by commas, and are NOT enclosed in quotes, so we can split the string by commas
            else:
                for j in range(rows):
                    try:
                        values += (
                            df[col_name]
                            .iloc[j]
                            .strip()
                            .replace("[", "")
                            .replace("]", "")
                            .split(",")
                        )
                    except KeyError:
                        continue

            # Remove spaces from the values
            values = [value.strip() for value in values]
            # Remove duplicates from the list of keys
            values = list(set(values))

            # print(f"Column {col_name} has {rows} rows and {len(values)} unique values")

            # If the number of unique values is greater than 25, then we skip the column
            if len(values) > 25 or len(values) == 0:
                continue

            result_str += f"Column {col_name} is made of lists that can contain the following elements: {values}\n"

        return result_str


def process_row(row, column_selector):
    question = row["question"]
    dataset = utils.load_table(row["dataset"])

    return column_selector.select_relevant_columns(list(dataset.columns), question)


def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        pad_token_id=tokenizer.eos_token_id,
        quantization_config=BitsAndBytesConfig(
            llm_int8_enable_fp32_cpu_offload=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        torch_dtype="auto",
        device_map="auto",
    )

    column_selector = ColumnSelector(pipe)

    qa = utils.load_qa(name="semeval", split="dev")
    qa = pd.DataFrame(qa)
    qa = qa.head()

    qa["selected_columns"] = qa.apply(
        lambda row: process_row(row, column_selector), axis=1
    )

    print(qa.head())


if __name__ == "__main__":
    main()
