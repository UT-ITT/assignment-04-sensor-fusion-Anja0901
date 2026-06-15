import sys
import time
import random

import cv2
import numpy as np
import pyglet


def order_points(points):
    """Sortiert vier Eckpunkte in der Reihenfolge: tl, tr, br, bl."""
    pts = np.array(points, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def cv2_to_pyglet(image):
    """Erstellt ein Pyglet ImageData Objekt aus dem OpenCV-Bild."""
    height, width, channels = image.shape
    raw = image.tobytes()
    pitch = -width * channels
    return pyglet.image.ImageData(width, height, 'BGR', raw, pitch=pitch)


def find_board_corners(corners, ids):
    """Ermittelt die vier Markerzentren und sortiert sie als Board-Ecken."""
    if ids is None or len(ids) < 4:
        return None

    centers = []
    for marker_corners in corners:
        pts = marker_corners.reshape((4, 2))
        center = pts.mean(axis=0)
        centers.append(center)

    if len(centers) < 4:
        return None

    ordered = order_points(np.array(centers))
    return ordered


def get_inner_marker_corners(corners):
    """Ermittelt für jeden Marker-Eckpunkt, denjenigen, welcher zur Mitte des Markerboards zeigt."""
    marker_centers = [marker.reshape((4, 2)).mean(axis=0) for marker in corners]
    idxs = list(range(len(marker_centers)))

    # Verwende die gleiche Einordnung wie order_points, aber auf Markerzentren
    centers = np.array(marker_centers, dtype='float32')
    ordered_centers = order_points(centers)
    ordered_ids = []
    for target in ordered_centers:
        best = min(idxs, key=lambda i: np.linalg.norm(marker_centers[i] - target))
        ordered_ids.append(best)
        idxs.remove(best)

    points = []
    for idx, pos in zip(ordered_ids, ['tl', 'tr', 'br', 'bl']):
        pts = corners[idx].reshape((4, 2))
        if pos == 'tl':
            sel = pts[np.argmax(pts[:, 0] + pts[:, 1])]
        elif pos == 'tr':
            sel = pts[np.argmax(pts[:, 1] - pts[:, 0])]
        elif pos == 'br':
            sel = pts[np.argmin(pts[:, 0] + pts[:, 1])]
        else:  # bl
            sel = pts[np.argmax(pts[:, 0] - pts[:, 1])]
        points.append(sel)

    return np.array(points, dtype='float32')


def compute_game_area(corners, transform, width, height, margin=40):
    """Berechnet einen etwas kleineren Bereich innerhalb der Marker-Ecken."""
    inner_corners = get_inner_marker_corners(corners)
    warped_pts = cv2.perspectiveTransform(inner_corners.reshape(1, -1, 2), transform)[0]
    x_min = warped_pts[:, 0].min()
    y_min = warped_pts[:, 1].min()
    x_max = warped_pts[:, 0].max()
    y_max = warped_pts[:, 1].max()

    region_width = x_max - x_min
    region_height = y_max - y_min
    game_margin = min(margin, region_width * 0.1, region_height * 0.1)
    x0 = int(max(0, x_min + game_margin))
    y0 = int(max(0, y_min + game_margin))
    x1 = int(min(width - 1, x_max - game_margin))
    y1 = int(min(height - 1, y_max - game_margin))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


# Hautfarbene-Maske (bei Tageslicht, weißem Licht robuster zur Fingererkennung)
def find_skin_contour(warped_bgr):
    """Ermittelt die größte hautfarbene Kontur im Spielfeld"""
    ycrcb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2YCrCb)

    # typische Hautfarbintensitäten im YCrCb-Farbraum (https://www.researchgate.net/publication/290440563_Skin_Segmentation_Using_YCBCR_and_RGB_Color_Models)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, skin_mask
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    height, width = warped_bgr.shape[:2]
    if area < 1500 or area > width * height * 0.4:
        return None, skin_mask
    return contour, skin_mask


# Helligkeitsmaske (rubuster bei warmem Licht oder für andere dunkle Gegenstände, statt Finger)
def find_dark_contour(warped_gray,  game_area = None):
    """Ermittelt die größte dunkle Kontur im Spielfeld"""
    blurred = cv2.GaussianBlur(warped_gray, (7, 7), 0)
    _, dark_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Konturen nur innerhalb der game_area erkennen, da sonst bei leerem board immer die Marker erkannt werden
    if game_area is not None:
        x0, y0, x1, y1 = game_area

        allowed_mask = np.zeros_like(dark_mask)
        allowed_mask[y0:y1, x0:x1] = 255

        dark_mask = cv2.bitwise_and(dark_mask, allowed_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, dark_mask
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    height, width = warped_gray.shape[:2]
    if area < 1500 or area > width * height * 0.4:
        return None, dark_mask
    return contour, dark_mask


def point_inside_contour(contour, point):
    return cv2.pointPolygonTest(contour, point, False) >= 0


def random_target_position(width, height, contour=None, region=None, margin=40, max_attempts=200):
    """Erzeugt eine zufällige Position innerhalb des Spielfeldes, (außerhalb der erkannten Kontur)"""
    if region is not None:
        x0, y0, x1, y1 = region
        region_width = x1 - x0
        region_height = y1 - y0
        margin = int(min(margin, region_width * 0.1, region_height * 0.1))
        min_x = x0 + margin
        max_x = x1 - margin
        min_y = y0 + margin
        max_y = y1 - margin
        if max_x <= min_x or max_y <= min_y:
            region = None
    if region is None:
        min_x, min_y = margin, margin
        max_x, max_y = width - margin, height - margin

    for _ in range(max_attempts):
        x = random.randint(int(min_x), int(max_x))
        y = random.randint(int(min_y), int(max_y))
        if contour is None or not point_inside_contour(contour, (x, y)):
            return (x, y)
    return (random.randint(int(min_x), int(max_x)), random.randint(int(min_y), int(max_y)))


def reset_game(game_state, width, height):
    """Setzt den Spielzustand zurück."""
    game_state['points'] = 0
    game_state['start_time'] = None
    game_state['end_time'] = None
    game_state['game_over'] = False
    game_state['target'] = random_target_position(width, height)


# ------------------------------------------------------------------

def main():
    video_id = 0
    if len(sys.argv) > 1:
        try:
            video_id = int(sys.argv[1])
        except ValueError:
            print('Verwende Standardkamera 0')
            video_id = 0

    cap = cv2.VideoCapture(video_id)
    if not cap.isOpened():
        raise RuntimeError('Webcam konnte nicht geöffnet werden.')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        raise RuntimeError('Ungültige Webcam-Auflösung.')

    window = pyglet.window.Window(width=width, height=height, caption='AR Game', resizable=False)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    game_state = {
        'target': (width // 2, height // 2),
        'points': 0,
        'start_time': None,
        'end_time': None,
        'contour': None,
        'display_image': None,
        'mask_message': '',
        'warp_active': False,
        'target_radius': 18,
        'max_points': 20,
        'game_over': False,
        'use_dark_mask': False,
        'game_area': None,
    }

    game_state['target'] = random_target_position(width, height)

    def update(dt):
        ret, frame = cap.read()
        
        if not ret or frame is None:
            return
        

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) >= 4:
            board_corners = find_board_corners(corners, ids)
            if board_corners is not None:
                destination = np.array([
                    [0.0, 0.0],
                    [width - 1.0, 0.0],
                    [width - 1.0, height - 1.0],
                    [0.0, height - 1.0],
                ], dtype='float32')
                transform = cv2.getPerspectiveTransform(board_corners, destination)
                warped = cv2.warpPerspective(frame, transform, (width, height))
                game_state['warp_active'] = True
                game_area = compute_game_area(corners, transform, width, height)
                game_state['game_area'] = game_area
                if game_area is not None:
                    x0, y0, x1, y1 = game_area
                    tx, ty = game_state['target']
                    if not (x0 <= tx <= x1 and y0 <= ty <= y1):
                        game_state['target'] = random_target_position(width, height, region=game_area)
            else:
                warped = frame.copy()
                game_state['warp_active'] = False
                game_state['game_area'] = None
        else:
            warped = frame.copy()
            game_state['warp_active'] = False
            game_state['game_area'] = None

        warped_display = warped.copy()

        if game_state['warp_active']:
            if game_state['use_dark_mask']:
                contour, mask = find_dark_contour(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), game_state.get('game_area'))
                contour_color = (200, 200, 200)
                if contour is not None:
                    game_state['mask_message'] = 'Helligkeitsmaske aktiv'
                else:
                    game_state['mask_message'] = 'Keine Kontur mit Helligkeitsmaske erkannt'
            else:
                contour, mask = find_skin_contour(warped)
                contour_color = (0, 0, 255)
                if contour is not None:
                    game_state['mask_message'] = 'Hautfarben-Maske aktiv'
                else:
                    game_state['mask_message'] = 'Keine Kontur mit Hautfarbenmaske erkannt'
        else:
            contour = None
            game_state['mask_message'] = ''

        game_state['contour'] = contour

        if contour is not None:
            cv2.drawContours(warped_display, [contour], -1, contour_color, 3)
            cv2.drawContours(warped_display, [contour], -1, contour_color, 1)

        if game_state['warp_active']:
            tx, ty = game_state['target']
            cv2.circle(warped_display, (tx, ty), game_state['target_radius'], (0, 0, 255), -1)
            cv2.circle(warped_display, (tx, ty), game_state['target_radius'], (255, 255, 255), 2)

        # Bild spiegeln, sonst ist man ständig verwirrt beim Spielen weil alles falschrum ist
        warped_display = cv2.flip(warped_display, 1)

        if contour is not None and not game_state['game_over']:
            if point_inside_contour(contour, (tx, ty)):
                game_state['points'] += 1
                if game_state['start_time'] is None:
                    game_state['start_time'] = time.perf_counter()
                if game_state['points'] >= game_state['max_points']:
                    game_state['end_time'] = time.perf_counter() - game_state['start_time']
                    game_state['game_over'] = True
                else:
                    game_state['target'] = random_target_position(width, height, contour, region=game_state.get('game_area'))

        text_color = (255, 255, 255)
        cv2.putText(warped_display, f'Punkte: {game_state["points"]}/{game_state["max_points"]}',
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)

        if game_state['start_time'] is None:
            elapsed = 0.0
        elif game_state['end_time'] is None:
            elapsed = time.perf_counter() - game_state['start_time']
        else:
            elapsed = game_state['end_time']

        cv2.putText(warped_display, f'Zeit: {elapsed:.1f} s',
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.85, text_color, 2, cv2.LINE_AA)
        cv2.putText(warped_display, game_state['mask_message'],
                    (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

        if not game_state['warp_active']:
            cv2.putText(warped_display, 'Kein Board mit Aruco-Markern gefunden. Wenn du keins hast, schau mal in C205a vorbei:) ',
                        (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)

        if game_state['game_over']:
            game_over_text = 'Geschafft!'
            end_text = f'Zeit: {elapsed:.1f} s'
            restart_text = 'Neustart: Taste R'
            font = cv2.FONT_HERSHEY_DUPLEX
            scale_big = 1.5
            scale_small = 0.9
            thickness_big = 3
            thickness_small = 2

            size1, _ = cv2.getTextSize(game_over_text, font, scale_big, thickness_big)
            size2, _ = cv2.getTextSize(end_text, font, scale_small, thickness_small)
            size3, _ = cv2.getTextSize(restart_text, font, scale_small, thickness_small)

            x1 = (width - size1[0]) // 2
            y1 = height // 2 - 30
            x2 = (width - size2[0]) // 2
            y2 = height // 2 + 20
            x3 = (width - size3[0]) // 2
            y3 = height // 2 + 60

            cv2.putText(warped_display, game_over_text, (x1, y1), font, scale_big, (0, 0, 255), thickness_big, cv2.LINE_AA)
            cv2.putText(warped_display, end_text, (x2, y2), font, scale_small, (255, 255, 255), thickness_small, cv2.LINE_AA)
            cv2.putText(warped_display, restart_text, (x3, y3), font, scale_small, (255, 255, 255), thickness_small, cv2.LINE_AA)

         
        game_state['display_image'] = cv2_to_pyglet(warped_display)

    @window.event
    def on_draw():
        window.clear()
        if game_state['display_image'] is not None:
            game_state['display_image'].blit(0, 0)

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.R:
            reset_game(game_state, width, height)
        elif symbol == pyglet.window.key.M:
            game_state['use_dark_mask'] = not game_state['use_dark_mask']

    @window.event
    def on_close():
        cap.release()
        window.close()

    pyglet.clock.schedule_interval(update, 1 / 30.0)
    pyglet.app.run()


if __name__ == '__main__':
    main()
