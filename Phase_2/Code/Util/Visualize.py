import matplotlib.pyplot as plt
import numpy as np

# import SQL library
import sqlite3

connection = sqlite3.connect('../../database/Phase_2.db')
c = connection.cursor()

get_HoG_value = """SELECT BOF_HOG FROM data WHERE videoID = 10;"""
get_HoF_value = """SELECT BOF_HOF FROM data WHERE videoID = 1;"""

c.execute(get_HoG_value)
rows = c.fetchall()

cleaned_str = rows[0][0].strip("[]")
number_list = list(map(int, cleaned_str.split()))

data = np.array(number_list).reshape(12, 40)

def Visualize_HoG_HoF(data):
    tau_values = [2, 4]
    sigma_values = [4, 8, 16, 32, 64, 128]

    fig, axes = plt.subplots(3, 4, figsize=(12, 8))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    i = 0
    for t in tau_values:
        for s in sigma_values:
            axes[i].hist(data[i], bins=np.arange(41) - 0.5, edgecolor='black')
            axes[i].set_title(f'Histogram for tau={t} sigma={s}')
            i += 1

    # Adjust layout for better spacing
    plt.tight_layout()

    # Display the histograms
    plt.show()

def Visualize_HoG(videoID):
    print('x')

def Visualize_HoF(videoID):
    print('x')

Visualize_HoG_HoF()

connection.close()
