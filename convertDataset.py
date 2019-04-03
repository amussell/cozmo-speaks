import pandas as pd
import CS481Dataset

def convert(filename: str):
    # Add additional column indicating positive examples to an existing dataframe
    df = loadFromCSV(filename)
    df['annotation'] = 1
    
    # Randomly choose negative examples for each color equal to the number of positives,
    # and add them to the dataframe.
    for color in df.label.unique():
        negativeSamples = df[df.label != color].sample(len(df.label == color))
        negativeSamples['annotation'] = 0
        df.append(negativeSamples)