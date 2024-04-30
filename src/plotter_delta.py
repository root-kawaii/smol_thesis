import matplotlib.pyplot as plt
import numpy as np

# Generating some random data
np.random.seed(10)

x = [
    "15 - 10",
    "14 - 6",
    "13 - 11",
    "12 - 3",
    "11 - 14",
    "10 - 7",
    "9 - 0",
    "8 - 2",
    "7 - 15",
]


accuracy_2 = [
    0.8532,
    0.8485,
    0.8471,
    0.842,
    0.8367,
    0.832,
    0.8285,
    0.8123,
    0.8178,
    0.7361,
]
f1_score_2 = [
    0.8279,
    0.8242,
    0.817,
    0.8194,
    0.8154,
    0.8076,
    0.8069,
    0.7919,
    0.7963,
    0.7184,
]


std_data_ac_2 = [
    0.005128203,
    0.006858865,
    0.005365904,
    0.007093232,
    0.004287042,
    0.007487694,
    0.005618573,
    0.011471681,
    0.009159471,
    0.006235253,
]

std_data_f1_2 = [
    0.006044684,
    0.007350249,
    0.00705451,
    0.007617024,
    0.003320661,
    0.007184426,
    0.005864662,
    0.009475886,
    0.012247129,
    0.007207906,
]


accuracy_2 = [x * 100 for x in accuracy_2]
f1_score_2 = [x * 100 for x in f1_score_2]
std_data_ac_2 = [x * 100 for x in std_data_ac_2]
std_data_f1_2 = [x * 100 for x in std_data_f1_2]

plot_data = []
plot_data_2 = []

for i, j in enumerate(accuracy_2):
    if i != 0:
        plot_data.append(
            accuracy_2[0] + std_data_ac_2[0] - accuracy_2[i] - std_data_ac_2[i]
        )
    if i != 0:
        plot_data_2.append(
            f1_score_2[0] + std_data_f1_2[0] - f1_score_2[i] - std_data_f1_2[i]
        )


plt.plot(
    plot_data,
    label="Accuracy",
)
plt.plot(x, plot_data_2, label="F1-Score")

# # Creating a box plot
# plt.errorbar(
#     x,
#     accuracy_2,
#     yerr=std_data_ac_2,
#     fmt="-o",
#     color="tab:orange",
#     ecolor="lightgray",
#     elinewidth=5,
#     capsize=0,
#     label="Accuracy",
# )

# plt.errorbar(
#     x,
#     f1_score_2,
#     yerr=std_data_f1_2,
#     fmt="-o",
#     color="tab:blue",
#     ecolor="lightgray",
#     elinewidth=5,
#     capsize=0,
#     label="F1-Score",
# )

for i in range(len(x)):
    plt.text(
        x[i],
        plot_data[i],
        f"{plot_data[i]:.1f}",
        fontsize=10,
        ha="left",
        va="bottom",
    )

for i in range(len(x)):
    plt.text(
        x[i],
        plot_data_2[i],
        f"{plot_data_2[i]:.1f}",
        fontsize=10,
        ha="left",
        va="bottom",
    )
# plt.gca().invert_xaxis()
# plt.xticks(x)
plt.axhline(y=5, color="red", linestyle="--", label="Threshold (5%)")

# Adding labels and title
plt.xlabel("Number of channels - {Removed channel}")
plt.ylabel("Metric difference with 16 channels (%)")
plt.title("Animal 3 - ENGNet100K - 100ms - 4 classes")
plt.legend()

plt.show()


"""
x = [
    "16 - 6",
    "15 - 10",
    "14 - 11",
    "13 - 7",
    "12 - 9",
    "11 - 5",
    "10 - 15",
    "9 - 14,
    "8 - 0",
]

accuracy_1 = [
    0.993,
    0.98901795,
    0.98475282,
    0.98042041,
    0.97565817,
    0.97050610,
    0.96736621,
    0.961954361,
    0.942852,
]

f1_1 = [
    0.991,
    0.9822098,
    0.9768506,
    0.9681791,
    0.9644779,
    0.95510062,
    0.9505889,
    0.9425473,
    0.9161532,
]



std_data_ac_1 = [
    0.002,
    0.002499003,
    0.002869618,
    0.003067728,
    0.005147021,
    0.005147721,
    0.003418154,
    0.002530038,
    0.004181659,
]


std_data_f1_1 = [
    0.002,
    0.004389057,
    0.003043836,
    0.006778012,
    0.008127111,
    0.009083839,
    0.006526623,
    0.004209037,
    0.00914046,
]




















x = [
    "16 - 10",
    "15 - 6",
    "14 - 11",
    "13 - 3",
    "12 - 14",
    "11 - 7",
    "10 - 0",
    "9 - 2,
    "8 - 15",
]


data = [0.8532, 0.8485, 0.8471, 0.842, 0.8367, 0.832, 0.8285, 0.8123, 0.8178, 0.7361]
data_2 = [0.8279, 0.8242, 0.817, 0.8194, 0.8154, 0.8076, 0.8069, 0.7919, 0.7963, 0.7184]


std_data = [
    0.005128203,
    0.006858865,
    0.005365904,
    0.007093232,
    0.004287042,
    0.007487694,
    0.005618573,
    0.011471681,
    0.009159471,
    0.006235253,
]

std_data_2 = [
    0.006044684,
    0.007350249,
    0.00705451,
    0.007617024,
    0.003320661,
    0.007184426,
    0.005864662,
    0.009475886,
    0.012247129,
    0.007207906,
]"""
