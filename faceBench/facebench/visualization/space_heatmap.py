import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go
import numpy as np
from typing import Optional


def plot_face_3d_error(
        vertices: np.ndarray,
        errors: np.ndarray,
        title: Optional[str] = None,
        cmap: str = "jet",
        elev: int = 30,
        azim: int = -75,
        figsize: tuple = (10, 10),
        vmax: Optional[float] = None,
        save_path: Optional[str] = None
):
    """
    Visualizza una mesh 3D con errori per-vertice colorati come heatmap.

    Parameters
    ----------
    vertices : (N, 3)
        Vertici della mesh.
    errors : (N,)
        Errori associati ai vertici (in mm).
    title : str, optional
        Titolo della figura.
    cmap : str
        Colormap per l’errore.
    elev : int
        Elevazione della camera.
    azim : int
        Azimut della camera.
    figsize : tuple
        Dimensione della figura.
    vmax : float, optional
        Valore massimo per normalizzazione del colore.
    save_path : str, optional
        Se fornito, salva l’immagine a questo path.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    if vmax is None:
        vmax = np.percentile(errors, 95)
    norm = plt.Normalize(vmin=0, vmax=vmax)
    colors_map = cm.get_cmap(cmap)(norm(errors))

    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
               c=colors_map, s=1, linewidth=0)

    ax.view_init(elev=elev, azim=azim)
    ax.axis('off')
    if title:
        ax.set_title(title)

    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    fig.colorbar(mappable, shrink=0.5)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()


def plot_face_3d_error_interactive(
        vertices: np.ndarray,
        errors: np.ndarray,
        title: Optional[str] = "3D Face Error Heatmap",
        colormap: str = "Jet",
        point_size: float = 2.0,
        vmax: Optional[float] = None,
        save_html: Optional[str] = None
):
    """
    Visualizza una mesh 3D con errori per-vertice in modo interattivo.

    Parameters
    ----------
    vertices : (N, 3)
        Vertici della mesh.
    errors : (N,)
        Errori associati ai vertici (in mm).
    title : str
        Titolo della visualizzazione.
    colormap : str
        Colormap (Plotly supporta "Jet", "Viridis", ecc.).
    point_size : float
        Dimensione dei punti nel 3D scatter.
    vmax : float, optional
        Valore massimo per normalizzazione del colore.
    save_html : str, optional
        Path per salvare in HTML. Se None, mostra interattivamente.
    """
    if vmax is None:
        vmax = np.percentile(errors, 95)

    fig = go.Figure(data=[go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color=np.clip(errors, 0, vmax),
            colorscale=colormap,
            colorbar=dict(title="Error (mm)"),
            cmin=0,
            cmax=vmax,
            opacity=0.9
        )
    )])

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        showlegend=True
    )

    if save_html:
        fig.write_html(save_html)
        print(f"✅ Saved to {save_html}")
    else:
        fig.show()


def plot_point_clouds_with_error_interactive(
        R: np.ndarray,
        G: np.ndarray,
        errors: np.ndarray,
        title: str = "3D Heatmap",
        colormap: str = "Jet",
        size: int = 1,
        vmax: Optional[float] = None,
        save_path: Optional[str] = None
):
    """
    Visualizza la mesh ricostruita (R) e ground-truth (G) in 3D con errori proiettati come colormap.

    Parameters
    ----------
    R : (N, 3) np.ndarray
        Vertici della mesh ricostruita (già allineata).
    G : (N, 3) np.ndarray
        Vertici della mesh ground-truth.
    errors : (N,) np.ndarray
        Errori per punto (in mm).
    title : str
        Titolo della figura.
    colormap : str
        Colormap da usare (es. "Jet", "Viridis", "Plasma").
    size : int
        Dimensione dei punti.
    vmax : float, optional
        Valore massimo per normalizzazione del colore.
    save_path : str, optional
        Se specificato, salva il grafico come file HTML interattivo.
    """
    if vmax is None:
        vmax = np.percentile(errors, 95)

    fig = go.Figure()

    # Mesh Ricostruita con colori per errore
    fig.add_trace(go.Scatter3d(
        x=R[:, 0], y=R[:, 1], z=R[:, 2],
        mode='markers',
        marker=dict(
            size=size,
            color=np.clip(errors, 0, vmax),
            colorscale=colormap.lower(),
            colorbar=dict(title="Error (mm)"),
            cmin=0,
            cmax=vmax,
            opacity=0.7
        ),
        name="Reconstructed"
    ))

    # Mesh Ground-truth
    fig.add_trace(go.Scatter3d(
        x=G[:, 0], y=G[:, 1], z=G[:, 2],
        mode='markers',
        marker=dict(
            size=size,
            color='black',
            opacity=0.5,
            symbol='square'
        ),
        name="Ground-truth"
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data'
        ),
        showlegend=True
    )

    if save_path:
        fig.write_html(save_path)
        print(f"🌐 Saved interactive plot to {save_path}")
    else:
        fig.show()
