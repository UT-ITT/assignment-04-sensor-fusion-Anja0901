

import sys
import time
import numpy as np
import cv2
import pyglet


from DIPPID import SensorUDP




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
    """Ermittelt für jeden Marker-Eckpunkt denjenigen, der zur Mitte des Boards zeigt."""
    marker_centers = [marker.reshape((4, 2)).mean(axis=0) for marker in corners]
    idxs = list(range(len(marker_centers)))

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


def transform_point(point, transform):
    """Transformiert einen Punkt mittels Perspektivtransformation."""
    pts = np.array([[point]], dtype='float32')
    transformed = cv2.perspectiveTransform(pts, transform)
    return transformed[0][0]


# --- Komplementärer Filter --------------------------------------------------
class ComplementaryFilter:
    """Komplementärer Filter zur Fusion von Kamera- und Beschleunigungsdaten."""

    def __init__(self, alpha=0.9):
        self.alpha = float(alpha)
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.last_time = time.perf_counter()
        self.accel_scale = 5.0

    def update(self, camera_pos, accel_x, accel_y, dt=None):
        """Aktualisiert Filter und gibt die neue Position zurück.
        """
        if dt is None:
            now = time.perf_counter()
            dt = now - self.last_time
            self.last_time = now
        else:
            self.last_time = time.perf_counter()

        dt = max(1e-3, min(dt, 0.1))

        # Geschwindigkeit integrieren
        self.velocity[0] += accel_x * self.accel_scale * dt
        self.velocity[1] += accel_y * self.accel_scale * dt

        # Vorhersage
        prediction = self.position + self.velocity * dt

        if camera_pos is None:
            alpha_eff = 0.0
            cam = self.position
        else:
            alpha_eff = self.alpha
            cam = np.array(camera_pos, dtype=float)

        # Update Position
        self.position = alpha_eff * cam + (1.0 - alpha_eff) * prediction
        return tuple(self.position.astype(int))

    def reset(self, initial_pos):
        self.position = np.array(initial_pos, dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.last_time = time.perf_counter()

    def set_alpha(self, alpha):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))



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

    window = pyglet.window.Window(width=width, height=height,
                                  caption='Sensor Fusion', resizable=False)

   
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

   
    dippid = None
    if SensorUDP is not None:
        try:
            dippid = SensorUDP(5700)
        except Exception:
            pass

    comp_filter = ComplementaryFilter(alpha=0.9)

    state = {
        'display_image': None,
        'transform': None,
        'warp_active': False,
        'camera_pos': None,
        'predicted_pos': None,
        'alpha': 0.9,
        'message': 'Bitte zuerst nur das Board halten zur Kalibrierung.',
        'accel_x': 0.0,
        'accel_y': 0.0,
        'accel_z': 0.0,
        'calibrated': False,
        'board_ids': set(),
    }

    
    def on_button_1(val):
        if val and state['camera_pos'] is not None:
            comp_filter.reset(state['camera_pos'])
            state['message'] = 'Filter zurückgesetzt'

    if dippid is not None:
        try:
            def accel_callback(v):
                state['accel_x'] = v['x']
                state['accel_y'] = v['y']
                state['accel_z'] = v['z']

            dippid.register_callback('accelerometer', accel_callback)
            dippid.register_callback('button_1', on_button_1)
        except Exception:
            pass

    
    def update(dt):
        ret, frame = cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        state['camera_pos'] = None

        if ids is not None and len(ids) >= 4:
            board_corners = find_board_corners(corners, ids)
            if board_corners is not None:
                dest = np.array([[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]], dtype='float32')
                transform = cv2.getPerspectiveTransform(board_corners, dest)
                state['transform'] = transform
                warped = cv2.warpPerspective(frame, transform, (width, height))
                state['warp_active'] = True

                ids_flat = ids.flatten().tolist()
                # Kalibrierung: wenn noch nicht kalibriert und mindestens 4 Marker sichtbar
                if (not state['calibrated']) and len(ids_flat) >= 4:
                    state['board_ids'] = set([int(x) for x in ids_flat])
                    state['calibrated'] = True
                    state['message'] = f"Board kalibriert: IDs={sorted(state['board_ids'])}"

                # Wenn kalibriert: finde ersten Marker, dessen ID nicht im Board ist
                if state['calibrated']:
                    phone_idx = None
                    phone_id = None
                    for i, m in enumerate(ids_flat):
                        if int(m) not in state['board_ids']:
                            phone_idx = i
                            phone_id = int(m)
                            break
                    if phone_idx is not None:
                        mc = corners[phone_idx].reshape((4, 2))
                        center = mc.mean(axis=0)
                        transformed = transform_point(center, state['transform'])

                        try:
                           
                            board_corner_indices = [i for i, m_id in enumerate(ids_flat) if int(m_id) in state['board_ids']]
                            board_corners = np.array([corners[i] for i in board_corner_indices])
                            inner = get_inner_marker_corners(board_corners)
                            warped_inner = cv2.perspectiveTransform(inner.reshape(1, -1, 2), state['transform'])[0]
                            x_min = warped_inner[:, 0].min()
                            x_max = warped_inner[:, 0].max()
                            y_min = warped_inner[:, 1].min()
                            y_max = warped_inner[:, 1].max()
                            tx, ty = transformed[0], transformed[1]
                            if (x_min <= tx <= x_max) and (y_min <= ty <= y_max):
                                state['camera_pos'] = transformed
                                state['message'] = f"Marker {phone_id} erkannt | alpha={state['alpha']:.2f}"
                            else:
                                state['camera_pos'] = None
                        except Exception:
                            state['camera_pos'] = transformed
                            state['message'] = f'Marker {phone_id} erkannt | alpha={state['alpha']:.2f}'
            else:
                warped = frame.copy()
                state['warp_active'] = False
        else:
            warped = frame.copy()
            state['warp_active'] = False

        # Filter-Update
        state['predicted_pos'] = comp_filter.update(
            state['camera_pos'],
            state['accel_x'],
            state['accel_y'],
            dt
        )

        # Anzeige spiegeln
        disp = cv2.flip(warped, 1)

        # Kamera-Punkt (rot)
        if state['camera_pos'] is not None:
            fx = width - int(state['camera_pos'][0])
            fy = int(state['camera_pos'][1])
            cv2.circle(disp, (fx, fy), 12, (0, 0, 255), -1)
            cv2.circle(disp, (fx, fy), 12, (255, 255, 255), 2)

        # Vorhersage-Punkt (grün)
        if state['predicted_pos'] is not None:
            fx = width - int(state['predicted_pos'][0])
            fy = int(state['predicted_pos'][1])
            cv2.circle(disp, (fx, fy), 10, (0, 255, 0), -1)
            cv2.circle(disp, (fx, fy), 10, (255, 255, 255), 2)

        # Textinfos
        cv2.putText(disp, state['message'], (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        accel_msg = f"Accel: x={state['accel_x']:.2f} y={state['accel_y']:.2f}"
        cv2.putText(disp, accel_msg, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, "Rot=Kamera  Gruen=Vorhersage  Pfeile: alpha anpassen", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        state['display_image'] = cv2_to_pyglet(disp)

    @window.event
    def on_draw():
        window.clear()
        if state['display_image'] is not None:
            state['display_image'].blit(0, 0)

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.UP:
            state['alpha'] = min(1.0, state['alpha'] + 0.05)
            comp_filter.set_alpha(state['alpha'])
            state['message'] = f"alpha angepasst: {state['alpha']:.2f}"
        elif symbol == pyglet.window.key.DOWN:
            state['alpha'] = max(0.0, state['alpha'] - 0.05)
            comp_filter.set_alpha(state['alpha'])
            state['message'] = f"alpha angepasst: {state['alpha']:.2f}"

    @window.event
    def on_close():
        cap.release()
        if dippid is not None:
            try:
                dippid.disconnect()
            except Exception:
                pass
        window.close()

    pyglet.clock.schedule_interval(update, 1 / 30.0)
    pyglet.app.run()


if __name__ == '__main__':
    main()
