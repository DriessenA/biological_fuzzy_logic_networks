# from matplotlib.colors import rgb2hex, hex2color
import plotly.express as px

# CMAP of 24 colors
cmap_24 = px.colors.qualitative.Dark24

default_hex = [
    "#004D40",  # Dark blue/green
    "#0072B2",  # Blue
    "#D55E00",  # orange
    "#F0E442",  # Yellow
    "#56B4E9",  # Lightblue
    "#5BD527",  # Green
    "#933CEE",  # Purple
    "#C84A8D",  # Pink
]

default_colors = {
    "darkblue": "#0072B2",
    "blue": "#004D40",
    "orange": "#D55E00",
    "yellow": "#F0E442",
    "lightblue": "#56B4E9",
    "green": "#5BD527",
    "purple": "#933CEE",
    "pink": "#C84A8D",
}

perturbation_models_dict = {
    "student_division_same_input": "#F0E442",  # Yellow
    "student_division_random_input": "#D55E00",  # Orange
    "teacher_division_same_input": "#56B4E9",  # Lightblue
    "teacher_division_random_input": "#0072B2",  # Blue
    "untrained_division_same_input": "#C84A8D",  # Pink
    "untrained_division_random_input": "#933CEE",  # Purple
}

models_dict = {
    "student_same_input": "#F0E442",  # Yellow
    "student_random_input": "#D55E00",  # Orange
    "teacher_same_input": "#56B4E9",  # Lightblue
    "teacher_random_input": "#0072B2",  # Blue
    "untrained_same_input": "#C84A8D",  # Pink
    "untrained_random_input": "#933CEE",  # Purple
}
