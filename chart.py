import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis("off")

# Define box style
box_style = dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="lightblue")

# Define positions for flow chart elements
positions = {
    "start": (0.5, 0.9),
    "upload": (0.5, 0.75),
    "task_id": (0.5, 0.6),
    "poll": (0.5, 0.45),
    "processed": (0.5, 0.3),
    "display": (0.5, 0.15),
}

# Add text boxes
elements = {
    "start": "Start",
    "upload": "Upload Image",
    "task_id": "Receive Task ID",
    "poll": "Poll Task Status",
    "processed": "Image Processed?",
    "display": "Display Result",
}

for key, text in elements.items():
    ax.text(*positions[key], text, ha="center", va="center", fontsize=10,
            bbox=box_style)

# Draw arrows
arrowprops = dict(arrowstyle="->", lw=1.5, color="black")

arrows = [
    ("start", "upload"),
    ("upload", "task_id"),
    ("task_id", "poll"),
    ("poll", "processed"),
    ("processed", "display"),
]

for src, dst in arrows:
    ax.annotate("", xy=positions[dst], xytext=positions[src], arrowprops=arrowprops)

# Show diagram
plt.show()
