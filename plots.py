import matplotlib.pyplot as plt

class Plots:

    @staticmethod
    def hormones(df):

        plt.figure(figsize=(12, 6))

        plt.plot(df["tick"], df["dopamine"], label="dopamine")
        plt.plot(df["tick"], df["cortisol"], label="cortisol")
        plt.plot(df["tick"], df["serotonin"], label="serotonin")
        plt.plot(df["tick"], df["melatonin"], label="melatonin")

        plt.legend()
        plt.title("Hormones")
        plt.grid()
        plt.show()

    @staticmethod
    def energy(df):

        plt.figure(figsize=(12, 4))
        plt.plot(df["tick"], df["energy"])
        plt.title("Energy")
        plt.grid()
        plt.show()

    @staticmethod
    def environment(df):

        plt.figure(figsize=(12, 4))
        plt.plot(df["tick"], df["food_available"], label="food")
        plt.plot(df["tick"], df["temperature"], label="temperature")
        plt.legend()
        plt.title("Environment")
        plt.grid()
        plt.show()