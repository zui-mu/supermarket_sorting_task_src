#!/usr/bin/env python3
"""Can this container open an X11 window with a usable OpenGL context?

Run with DISPLAY pointing at the host's X server, e.g.
    docker run --rm -e DISPLAY=host.docker.internal:0 ...
"""
from __future__ import annotations

import sys


def main() -> int:
    print(f"DISPLAY={__import__('os').environ.get('DISPLAY')!r}")

    try:
        import glfw
    except Exception as exc:
        print(f"glfw import failed: {exc}")
        return 2

    if not glfw.init():
        print("glfw.init() FAILED - no usable X11/OpenGL display")
        return 1
    print("glfw.init() ok")

    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    try:
        window = glfw.create_window(400, 300, "DSH display probe", None, None)
    except Exception as exc:
        print(f"create_window raised: {exc}")
        glfw.terminate()
        return 1
    if not window:
        print("create_window FAILED")
        glfw.terminate()
        return 1
    print("create_window ok")

    glfw.make_context_current(window)
    import OpenGL.GL as gl

    print(f"GL_VENDOR   = {gl.glGetString(gl.GL_VENDOR)}")
    print(f"GL_RENDERER = {gl.glGetString(gl.GL_RENDERER)}")
    print(f"GL_VERSION  = {gl.glGetString(gl.GL_VERSION)}")

    version = gl.glGetString(gl.GL_VERSION) or b""
    try:
        major, minor = (int(p) for p in version.split()[0].split(b".")[:2])
    except Exception:
        major, minor = 0, 0
    print(f"parsed GL {major}.{minor} (MuJoCo needs >= 2.1, gsplat needs >= 3.3)")

    for _ in range(5):
        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.destroy_window(window)
    glfw.terminate()
    print("RESULT: window opened and rendered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
