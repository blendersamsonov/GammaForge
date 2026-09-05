"""Opt-in real browser smoke test: GAMMAFORGE_BROWSER_TEST=1 pytest -q ... ."""

import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("GAMMAFORGE_BROWSER_TEST") != "1",
                                reason="opt-in browser test (requires Chromium and Playwright)")


def test_browser_layout_and_calculation(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    browser_path = shutil.which("chromium") or shutil.which("chromium-browser")
    if not browser_path:
        pytest.skip("Chromium is not installed")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    root = Path(__file__).resolve().parents[1]
    server_env = dict(os.environ)
    server_env.pop("PYTEST_CURRENT_TEST", None)
    with (tmp_path / "server.log").open("w+") as log:
        server = subprocess.Popen([sys.executable, "-m", "gammaforge.gui", "--no-browser", "--port", str(port)],
                                  cwd=root, stdout=log, stderr=subprocess.STDOUT, env=server_env)
        try:
            url = f"http://127.0.0.1:{port}/"
            for _ in range(100):
                try:
                    with urlopen(url, timeout=1) as response:
                        assert response.status == 200
                    break
                except OSError:
                    if server.poll() is not None:
                        log.seek(0)
                        pytest.fail(log.read())
                    time.sleep(0.1)
            else:
                log.seek(0)
                pytest.fail(f"Server did not start: {log.read()}")
            with playwright.sync_playwright() as p:
                browser = p.chromium.launch(executable_path=browser_path, headless=True,
                                            args=["--no-sandbox", "--enable-unsafe-swiftshader"])
                page = browser.new_page(viewport={"width": 1600, "height": 1200})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url)
                expect = playwright.expect
                expect(page.get_by_text("GammaForge", exact=True)).to_be_visible()
                expect(page.get_by_text("Updating estimates…")).to_have_count(0, timeout=20000)
                columns = page.locator(".gf-input-column")
                expect(columns).to_have_count(3)
                boxes = [columns.nth(i).bounding_box() for i in range(3)]
                assert max(b["height"] for b in boxes) - min(b["height"] for b in boxes) < 3
                assert max(b["y"] for b in boxes) - min(b["y"] for b in boxes) < 3
                for title in ("Target and outputs", "Analytical estimates", "Calculation engines"):
                    expect(page.get_by_text(title, exact=True)).to_have_count(1)
                row_positions = [page.get_by_text(title, exact=True).bounding_box()["y"]
                                 for title in ("Target and outputs", "Analytical estimates", "Calculation engines")]
                assert boxes[0]["y"] + boxes[0]["height"] < row_positions[0]
                assert row_positions == sorted(row_positions)
                page.screenshot(path=str(tmp_path / "inputs.png"), full_page=True)

                def field(key):
                    return page.locator(f'input[data-field="{key}"], [data-field="{key}"] input').first

                field("beam.bunch_charge").fill("invalid")
                expect(page.get_by_role("button", name="Calculate", exact=True)).to_be_disabled()
                field("beam.bunch_charge").fill("100")
                expect(page.get_by_role("button", name="Calculate", exact=True)).to_be_enabled()
                for key, value in {"sampling.n_particles": "64", "engine:xigma.n_steps": "12",
                                   "engine:xigma.n_bins_gamma": "4", "engine:xigma.n_bins_theta_x": "4",
                                   "engine:xigma.n_bins_theta_y": "4", "engine:xigma.n_bins_a0_shape": "8",
                                   "engine:xigma.n_bins_ahat": "4", "outputs.SPECTRUM.resolution.0": "16"}.items():
                    field(key).fill(value)
                page.get_by_role("button", name="Calculate", exact=True).click()
                expect(page.get_by_text("xigma: completed", exact=True)).to_be_visible(timeout=60000)
                page.get_by_role("tab", name="Results", exact=True).click()
                expect(page.get_by_text("Calculate to populate results.")).to_have_count(0)
                page.get_by_role("tab", name="Spectrum", exact=True).click()
                expect(page.locator(".js-plotly-plot:visible")).to_have_count(1)
                hdf5_button = page.get_by_role("button", name="Download xigma HDF5", exact=True).filter(visible=True)
                expect(hdf5_button).to_have_count(1)
                with page.expect_download() as download:
                    hdf5_button.click()
                from gammaforge.io.formats.hdf5 import load_results
                from gammaforge.io.target import OutputKind
                assert OutputKind.SPECTRUM.value in load_results(download.value.path()).photon_slices
                with page.expect_download() as download:
                    page.get_by_role("button", name="PNG", exact=True).click()
                assert Path(download.value.path()).read_bytes().startswith(b"\x89PNG")
                page.screenshot(path=str(tmp_path / "results.png"), full_page=True)
                page.get_by_role("switch", name="Split view").click()
                expect(page.locator(".gf-pane")).to_have_count(2)
                page.locator(".gf-pane").first.get_by_role("tab", name="Inputs", exact=True).click()
                expect(field("sampling.n_particles")).to_have_value("64")
                page.screenshot(path=str(tmp_path / "split.png"), full_page=True)
                page.get_by_role("switch", name="Split view").click()
                page.set_viewport_size({"width": 600, "height": 1000})
                expect(page.locator(".gf-pane")).to_have_count(1)
                expect(columns).to_have_count(3)
                expect(columns.nth(2)).to_be_visible()
                boxes = [columns.nth(i).bounding_box() for i in range(3)]
                assert boxes[0]["y"] < boxes[1]["y"] < boxes[2]["y"]
                assert all(b["x"] + b["width"] <= 600 for b in boxes)
                page.screenshot(path=str(tmp_path / "narrow.png"), full_page=True)
                assert not errors, errors
                browser.close()
        finally:
            server.terminate()
            server.wait(timeout=10)
