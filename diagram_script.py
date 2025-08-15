import matplotlib.pyplot as plt

def simple_flowchart():
    fig, ax = plt.subplots(figsize=(6,5))
    ax.axis('off')

    boxes = {
        "A": "Frontend\n(izbor modela i dataseta)",
        "B": "Flask backend\n(provjera validnosti)",
        "C": "Provjera cache fajla\n(već trenirano?)",
        "D": "Učitavanje i obrada podataka",
        "E": "Trening modela\n(s unaprijed testiranim\nhiperparametrima)",
        "F": "Generisanje rezultata\n(classification report,\ngrafici, vrijeme)",
        "G": "Slanje rezultata frontendu"
    }

    positions = {
        "A": (0.5, 0.9),
        "B": (0.5, 0.75),
        "C": (0.5, 0.6),
        "D": (0.3, 0.45),
        "E": (0.5, 0.3),
        "F": (0.7, 0.15),
        "G": (0.5, 0.0)
    }

    for key, text in boxes.items():
        x, y = positions[key]
        ax.text(x, y, text, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="black", lw=1.5))

    # Strelice (linije)
    def draw_arrow(start, end, label=None, offset=0.05):
        x_start, y_start = positions[start]
        x_end, y_end = positions[end]

        # Pomjerimo y koordinate da strelica ne ide direktno iz centra kutije
        if y_start > y_end:
            y_start -= offset
            y_end += offset
        else:
            y_start += offset
            y_end -= offset

        ax.annotate("",
                    xy=(x_end, y_end), xycoords='axes fraction',
                    xytext=(x_start, y_start), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="->", lw=1.5))
        if label:
            mid_x = (x_start + x_end) / 2
            mid_y = (y_start + y_end) / 2
            ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=9, backgroundcolor="white")


    draw_arrow("A", "B")
    draw_arrow("B", "C")
    draw_arrow("C", "G", label="Da")
    draw_arrow("C", "D", label="Ne")
    draw_arrow("D", "E")
    draw_arrow("E", "F")
    draw_arrow("F", "G")

    plt.title("Jednostavan dijagram toka", fontsize=14)
    plt.tight_layout()
    plt.show()


def complex_flowchart():
    fig, ax = plt.subplots(figsize=(10, 7))  # Šire i niže
    ax.axis('off')

    boxes = {
        "A": "Frontend\n- izbor modela\n- izbor dataseta",
        "B": "Flask backend\nProvjera validnosti zahtjeva",
        "C": "Provjera kombinacije\nmodel+dataset u cache-u",
        "D": "Ako postoji JSON → učitaj i pošalji",
        "E": "Učitavanje CSV iz\nHuggingFace",
        "F": "Preprocesiranje podataka",
        "G": "Podjela na train/test\n(stratifikovano)",
        "H": "Trening modela s\nhiperparametrima",
        "I": "Generisanje evaluacija\n(classification report,\nbar plot, log-loss, vrijeme)",
        "J": "Serijalizacija rezultata\nu JSON za cache",
        "K": "Slanje responsa frontendu"
    }

    positions = {
        "A": (0.5, 1.0),
        "B": (0.5, 0.85),
        "C": (0.5, 0.7),
        "D": (0.3, 0.5),
        "E": (0.7, 0.5),
        "F": (0.7, 0.4),
        "G": (0.7, 0.3),
        "H": (0.7, 0.2),
        "I": (0.7, 0.1),
        "J": (0.7, 0.0),
        "K": (0.5, -0.1)
    }


    for key, text in boxes.items():
        x, y = positions[key]
        ax.text(x, y, text, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="black", lw=1.5))

    def draw_arrow(start, end, label=None, offset=0.05):
        x_start, y_start = positions[start]
        x_end, y_end = positions[end]

        # Pomjerimo y koordinate da strelica ne ide direktno iz centra kutije
        if y_start > y_end:
            y_start -= offset
            y_end += offset
        else:
            y_start += offset
            y_end -= offset

        ax.annotate("",
                    xy=(x_end, y_end), xycoords='axes fraction',
                    xytext=(x_start, y_start), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="->", lw=1.5))
        if label:
            mid_x = (x_start + x_end) / 2
            mid_y = (y_start + y_end) / 2
            ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=9, backgroundcolor="white")


    # Povezivanje strelica
    draw_arrow("A", "B")
    draw_arrow("B", "C")
    draw_arrow("C", "D", label="Postoji")
    draw_arrow("C", "E", label="Ne postoji")
    draw_arrow("D", "K")
    draw_arrow("E", "F")
    draw_arrow("F", "G")
    draw_arrow("G", "H")
    draw_arrow("H", "I")
    draw_arrow("I", "J")
    draw_arrow("J", "K")

    plt.tight_layout()
    plt.show()


# Pokreni oba dijagrama
simple_flowchart()
complex_flowchart()
