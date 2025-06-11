import numpy as np
from scipy.interpolate import CubicSpline

def arc_length_parametrization(waypoints, num_samples):
    """
    Parametrisiert eine Trajektorie durch Bogenlänge statt durch Zeit.
    
    Args:
        waypoints: Array der Wegpunkte (n, 3) mit x,y,z Koordinaten
        num_samples: Anzahl der gewünschten Samples für die Ausgabe
        
    Returns:
        theta_values: Fortschrittsparameter (normalisierte Bogenlänge) [0,1]
        x_values: X-Koordinaten an den entsprechenden theta-Werten
        y_values: Y-Koordinaten an den entsprechenden theta-Werten
        z_values: Z-Koordinaten an den entsprechenden theta-Werten
    """
    print("Starting arc length parametrization...")
    
    # Erstelle Parameter für die ursprünglichen Wegpunkte (gleichmäßig verteilt)
    t_orig = np.linspace(0, 1, len(waypoints))
    
    # Erstelle kubische Splines durch die Wegpunkte
    cs_x = CubicSpline(t_orig, waypoints[:, 0])
    cs_y = CubicSpline(t_orig, waypoints[:, 1])
    cs_z = CubicSpline(t_orig, waypoints[:, 2])

    
    print(f"Created cubic splines through {len(waypoints)} waypoints")
    
    # Erzeuge eine dichte Abtastung für die Bogenlängenberechnung
    # Mehr Punkte = genauere Bogenlängenberechnung
    num_dense_samples = 1000
    t_dense = np.linspace(0, 1, num_dense_samples)
    
    # Berechne die Punkte auf der Spline für diese dichte Abtastung
    x_dense = cs_x(t_dense)
    y_dense = cs_y(t_dense)
    z_dense = cs_z(t_dense)
    points_dense = np.column_stack((x_dense, y_dense, z_dense))


    
    print(f"Created {num_dense_samples} dense samples for arc length calculation")
    



    # Berechne die Bogenlänge zwischen aufeinanderfolgenden Punkten
    # np.diff gibt die Differenz zwischen benachbarten Elementen zurück
    diffs = np.diff(points_dense, axis=0)  # Differenzen in x, y, z
    
    segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))  # Euklidische Distanzen

    # print(segment_lengths)
    # Berechne die kumulierte Bogenlänge bis zu jedem Punkt
    cumulative_length = np.zeros(num_dense_samples)
    cumulative_length[1:] = np.cumsum(segment_lengths)




    # Gesamtlänge der Trajektorie
    total_length = cumulative_length[-1]
    print(f"Total arc length of trajectory: {total_length:.4f} units")
    
    # Normalisiere die kumulierte Länge auf [0,1] für den Fortschrittsparameter theta
    theta_dense = cumulative_length / total_length
    
    # Erzeuge gleichmäßig verteilte theta-Werte für die Ausgabe
    theta_values = np.linspace(0, 1, num_samples)

    # Interpoliere t-Werte für die gleichmäßig verteilten theta-Werte
    # Dies ist der Schlüsselschritt: wir konvertieren von gleichmäßigen Bogenlängen
    # zurück zu den entsprechenden Parametern auf der ursprünglichen Spline
    t_interp = np.interp(theta_values, theta_dense, t_dense)


    print(f"Interpolated {num_samples} evenly spaced points along arc length")
    
    # Berechne die finalen x,y,z-Koordinaten für diese t-Werte
    x_values = cs_x(t_interp)
    y_values = cs_y(t_interp)
    z_values = cs_z(t_interp)

    # points_dense = np.column_stack((x_values, y_values, z_values))
    # diffs_1 = np.diff(points_dense, axis=0)  # Differenzen in x, y, z
    # segment_lengths_1 = np.sqrt(np.sum(diffs_1**2, axis=1))  # Euklidische Distanzen
    
    # print(f"Segment lengths after interpolation: {segment_lengths_1}")


    # Gib die Ergebnisse zurück
    return theta_values, x_values, y_values, z_values


def find_reference_points(theta, theta_values, x_des, y_des, z_des, lookahead=0.05):
    """
    Findet Referenzpunkte auf der Trajektorie für einen gegebenen Fortschrittsparameter theta
    und einen Lookahead-Wert.
    
    Args:
        theta: Der aktuelle Fortschrittsparameter (zwischen 0 und 1)
        theta_values: Array der Fortschrittsparameterwerte (normalisierte Bogenlänge) [0,1]
        x_des: Array der X-Koordinaten an den entsprechenden theta-Werten
        y_des: Array der Y-Koordinaten an den entsprechenden theta-Werten
        z_des: Array der Z-Koordinaten an den entsprechenden theta-Werten
        lookahead: Der Lookahead-Parameter für den nächsten Referenzpunkt (Default: 0.05)
        
    Returns:
        ref_point: Der aktuelle Referenzpunkt [x, y, z] für theta
        next_ref_point: Der nächste Referenzpunkt [x, y, z] für theta + lookahead
    """
    # Begrenze theta auf den gültigen Bereich [0, 1]
    theta = np.clip(theta, 0.0, 1.0)
    
    # Berechne die Indices der beiden nächstgelegenen Punkte
    # Der Fortschrittsparameter kann zwischen zwei Abtastpunkten liegen
    idx_low = np.searchsorted(theta_values, theta, side='right') - 1
    idx_high = min(idx_low + 1, len(theta_values) - 1)
    
    # Falls theta genau auf einem Abtastpunkt liegt
    if idx_low == idx_high or idx_low < 0:
        idx_low = max(0, idx_low)
        ref_point = np.array([x_des[idx_low], y_des[idx_low], z_des[idx_low]])
    else:
        # Lineare Interpolation zwischen den beiden nächstgelegenen Punkten
        t_low = theta_values[idx_low]
        t_high = theta_values[idx_high]
        
        # Falls die beiden theta-Werte identisch sind, keine Interpolation nötig
        if t_high == t_low:
            alpha = 0.0
        else:
            # Berechne den Interpolationsparameter
            alpha = (theta - t_low) / (t_high - t_low)
        
        # Interpoliere die Koordinaten
        x = x_des[idx_low] + alpha * (x_des[idx_high] - x_des[idx_low])
        y = y_des[idx_low] + alpha * (y_des[idx_high] - y_des[idx_low])
        z = z_des[idx_low] + alpha * (z_des[idx_high] - z_des[idx_low])
        
        ref_point = np.array([x, y, z])
    
    # Berechne den nächsten Referenzpunkt mit Lookahead
    next_theta = theta + lookahead
    # Stelle sicher, dass next_theta im gültigen Bereich [0, 1] liegt
    next_theta = np.clip(next_theta, 0.0, 1.0)
    
    # Berechne die Indices der beiden nächstgelegenen Punkte für next_theta
    idx_low_next = np.searchsorted(theta_values, next_theta, side='right') - 1
    idx_high_next = min(idx_low_next + 1, len(theta_values) - 1)
    
    # Falls next_theta genau auf einem Abtastpunkt liegt
    if idx_low_next == idx_high_next or idx_low_next < 0:
        idx_low_next = max(0, idx_low_next)
        next_ref_point = np.array([x_des[idx_low_next], y_des[idx_low_next], z_des[idx_low_next]])
    else:
        # Lineare Interpolation zwischen den beiden nächstgelegenen Punkten
        t_low_next = theta_values[idx_low_next]
        t_high_next = theta_values[idx_high_next]
        
        # Falls die beiden theta-Werte identisch sind, keine Interpolation nötig
        if t_high_next == t_low_next:
            alpha_next = 0.0
        else:
            # Berechne den Interpolationsparameter
            alpha_next = (next_theta - t_low_next) / (t_high_next - t_low_next)
        
        # Interpoliere die Koordinaten
        x_next = x_des[idx_low_next] + alpha_next * (x_des[idx_high_next] - x_des[idx_low_next])
        y_next = y_des[idx_low_next] + alpha_next * (y_des[idx_high_next] - y_des[idx_low_next])
        z_next = z_des[idx_low_next] + alpha_next * (z_des[idx_high_next] - z_des[idx_low_next])
        
        next_ref_point = np.array([x_next, y_next, z_next])
    
    return ref_point, next_ref_point

# also man hat jetzt x_values, y_values, z_values die quasi die Koordinaten entlang der Trajektorie angegeben;
# sodass die abstände der zwischen den Punkten immer gleich sind;

