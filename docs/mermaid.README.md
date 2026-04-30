# Mermaid workflow diagrams

The root [`README.md`](../README.md) embeds **thumbnail PNGs** for the two workflow diagrams so they render the same everywhere (including viewers that do not execute Mermaid). Each thumbnail links to a **higher-resolution PNG** (`*-large.png`) for detail when opened on GitHub.

## What the `.mmd` files are

Files ending in **`.mmd`** are **Mermaid diagram sources**: plain-text definitions (here, `flowchart LR` graphs) in the format expected by [Mermaid CLI](https://github.com/mermaid-js/mermaid-cli) (`mmdc`).

| Source (edit this) | README thumbnail | Full-size (click target in README) |
|--------------------|------------------|-------------------------------------|
| [`images/video-upload-workflow.mmd`](images/video-upload-workflow.mmd) | [`images/video-upload-workflow.png`](images/video-upload-workflow.png) | [`images/video-upload-workflow-large.png`](images/video-upload-workflow-large.png) |
| [`images/application-workflow.mmd`](images/application-workflow.mmd) | [`images/application-workflow.png`](images/application-workflow.png) | [`images/application-workflow-large.png`](images/application-workflow-large.png) |

**Workflow:** change the `.mmd` → regenerate **both** the thumbnail and large PNG for that diagram → commit so the README stays in sync.

## Regenerating PNGs (Podman + mermaid-cli)

From the **repository root**, using the `minlag/mermaid-cli` image.

### Thumbnails (shown inline in the README)

Defaults match Mermaid CLI (`800×600` viewport); `-b transparent` keeps a transparent background.

```bash
mkdir -p /tmp/mmd-out && chmod 777 /tmp/mmd-out

podman run --rm --user 0 \
  -v "$PWD/docs/images:/data:z" -v /tmp/mmd-out:/out:z \
  docker.io/minlag/mermaid-cli:latest \
  -i /data/video-upload-workflow.mmd -o /out/video-upload-workflow.png -b transparent

cp /tmp/mmd-out/video-upload-workflow.png docs/images/

podman run --rm --user 0 \
  -v "$PWD/docs/images:/data:z" -v /tmp/mmd-out:/out:z \
  docker.io/minlag/mermaid-cli:latest \
  -i /data/application-workflow.mmd -o /out/application-workflow.png -b transparent

cp /tmp/mmd-out/application-workflow.png docs/images/
```

### Large PNGs (linked when you click the thumbnail on GitHub)

Wider viewport plus Puppeteer scale for sharper output (adjust `-w`, `-H`, `-s` if you need even larger files):

```bash
mkdir -p /tmp/mmd-out && chmod 777 /tmp/mmd-out

podman run --rm --user 0 \
  -v "$PWD/docs/images:/data:z" -v /tmp/mmd-out:/out:z \
  docker.io/minlag/mermaid-cli:latest \
  -i /data/video-upload-workflow.mmd -o /out/video-upload-workflow-large.png \
  -b transparent -w 2400 -H 1600 -s 2

cp /tmp/mmd-out/video-upload-workflow-large.png docs/images/

podman run --rm --user 0 \
  -v "$PWD/docs/images:/data:z" -v /tmp/mmd-out:/out:z \
  docker.io/minlag/mermaid-cli:latest \
  -i /data/application-workflow.mmd -o /out/application-workflow-large.png \
  -b transparent -w 2400 -H 1600 -s 2

cp /tmp/mmd-out/application-workflow-large.png docs/images/
```

If you use **Node** locally, the same sources can be passed to `npx @mermaid-js/mermaid-cli` with equivalent `-i` / `-o` / `-w` / `-H` / `-s` arguments.

## Related assets

- [`images/architecture.svg`](images/architecture.svg) — separate static architecture diagram (not generated from these `.mmd` files).
- [`architecture-slides.html`](architecture-slides.html) — slide deck; see [`architecture-slides-README.md`](architecture-slides-README.md).
