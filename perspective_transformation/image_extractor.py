#!/usr/bin/env python3
import argparse 
import os
import sys

import cv2
import numpy as np


def parse_arguments():
    """Parst die Kommandozeilenparameter."""
    parser = argparse.ArgumentParser(
        description=(
            "Image Extractor"
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Pfad zum Eingabebild",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Pfad für das Ausgabebild",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Breite des Ausgabebildes",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Höhe des Ausgabebildes",
    )
    return parser.parse_args()


def compute_perspective_warp(image, points, width, height):
    """Berechnet das perspektivisch verzerrte Bild aus den ausgewählten Eckpunkten."""
    selected_points = np.array(points, dtype="float32")
    output_corners = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype="float32",
    )

    transform_matrix = cv2.getPerspectiveTransform(selected_points, output_corners)
    warped_img = cv2.warpPerspective(image, transform_matrix, (width, height))
    return warped_img


def draw_selection(image, selected_points):
    """Zeichnet ausgewählte Punkte und Linien auf eine Kopie des Bildes."""
    annotated = image.copy()
    for index, point in enumerate(selected_points):
        cv2.circle(annotated, tuple(point), 6, (0, 255, 0), -1)
        cv2.putText(
            annotated,
            str(index + 1),
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if len(selected_points) > 1:
        for i in range(len(selected_points) - 1):
            cv2.line(annotated, tuple(selected_points[i]), tuple(selected_points[i + 1]), (255, 0, 0), 2)

    if len(selected_points) == 4:
        cv2.line(annotated, tuple(selected_points[3]), tuple(selected_points[0]), (255, 0, 0), 2)

    return annotated


def main():
    args = parse_arguments()

    if not os.path.isfile(args.input):
        print(f"Fehler: Eingabedatei '{args.input}' wurde nicht gefunden.")
        sys.exit(1)

    image = cv2.imread(args.input)
    if image is None:
        print(f"Fehler: '{args.input}' konnte nicht geladen werden")
        sys.exit(1)

    # Hauptfenster
    window_name = "Bildauswahl"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    selected_points = []
    warped_image = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_points, warped_image

        if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
            selected_points.append((x, y))
            warped_image = None
            print(f"Punkt {len(selected_points)}: ({x}, {y}) ausgewählt")
            

    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        preview = draw_selection(image, selected_points)
        cv2.imshow(window_name, preview)

        if len(selected_points) == 4 and warped_image is None:
            warped_image = compute_perspective_warp(image, selected_points, args.width, args.height)
            # Perspektivisch verzerrtes Bild anzeigen
            cv2.imshow("Ergebnis", warped_image)
        

        key = cv2.waitKey(25) & 0xFF
        if key == 27:  # ESC
            if selected_points:
                selected_points = []
                warped_image = None
                cv2.destroyWindow("Ergebnis")
                print("Auswahl verworfen")
            else:
                print("Programm schließen")
                break
        elif key in (ord("s"), ord("S")) and warped_image is not None:
            cv2.imwrite(args.output, warped_image)
            print(f"Ergebnisbild gespeichert: {args.output}")

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
