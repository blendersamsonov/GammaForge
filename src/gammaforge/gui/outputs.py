def _render_collimated(available: dict[str, tuple[int, PhasespaceSlice]], view_state: dict) -> None:
    """Render collimated results with run selection checkboxes and view tabs."""
    import plotly.graph_objects as go
    from ..io.plotting import display_values, display_unit, _density_in_display_units, _density_label, _LABELS

    ui = _ui()

    all_names = list(available.keys())

    # Initialize selected runs in view_state
    if "collimated_selected_runs" not in view_state:
        view_state["collimated_selected_runs"] = all_names.copy()

    # Ensure selected runs are still valid
    view_state["collimated_selected_runs"] = [n for n in view_state["collimated_selected_runs"] if n in available]

    holder = ui.column().classes("w-full")

    # 1D views can be overlaid, 2D views show one at a time
    ONE_D_VIEWS = {"spectrum", "energy_at_theta_x_zero", "energy_at_theta_y_zero"}
    TWO_D_VIEWS = {"energy_theta_x", "energy_theta_y"}

    def redraw(selected_names: list[str]) -> None:
        holder.clear()
        with holder:
            if not selected_names:
                ui.label("Select at least one run to view.").classes("text-grey")
                return

            # Filter to only runs that have collimated spectrum
            valid_runs = {}
            for name in selected_names:
                if name in available:
                    run_id, slice_ = available[name]
                    try:
                        projections = collimated_projections(slice_)
                        valid_runs[name] = (run_id, projections)
                    except ValueError:
                        pass  # Skip runs that can't be projected

            if not valid_runs:
                ui.label("No selected runs have valid collimated spectrum data.").classes("text-grey")
                return

            valid_names = list(valid_runs.keys())

            # View tabs - collect all available views across all runs
            all_views = set()
            for name, (_, projections) in valid_runs.items():
                all_views.update(projections.keys())
            ordered_views = [v for v in ("energy_theta_x", "energy_theta_y", "energy_at_theta_x_zero", "energy_at_theta_y_zero", "spectrum") if v in all_views]

            current_view = view_state.get("collimated_view", "spectrum")
            if current_view not in ordered_views:
                current_view = ordered_views[0] if ordered_views else "spectrum"
            view_state["collimated_view"] = current_view

            def update_view(view):
                view_state["collimated_view"] = view
                plot_holder.clear()
                with plot_holder:
                    if view in ONE_D_VIEWS:
                        # Overlay all selected runs on the same axes
                        figure = None
                        for name in valid_names:
                            run_id, projections = valid_runs[name]
                            if view not in projections:
                                continue
                            slice_ = projections[view]
                            axes = slice_.axis_order
                            if not axes or len(axes) != 1:
                                continue

                            axis = axes[0]
                            x = display_values(axis, slice_.axes[axis])
                            y = _density_in_display_units(slice_.distr, axes)
                            color = _run_color(all_names.index(name))

                            if figure is None:
                                figure = go.Figure(
                                    go.Scatter(x=x, y=y, mode="lines", name=name,
                                               line={"color": color})
                                )
                                figure.update_layout(
                                    xaxis_title=f"{_LABELS[axis]} [{display_unit(axis)}]",
                                    yaxis_title=_density_label(axes),
                                    title=view.replace("_", " ").title(),
                                )
                            else:
                                figure.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name,
                                                           line={"color": color}))

                        if figure is None:
                            ui.label("No valid 1D data found.").classes("text-grey")
                        else:
                            _plot(figure)
                    else:
                        # 2D view - show dropdown to select one run
                        run_selector = ui.select(valid_names, value=valid_names[0], label="Run to display")

                        def update_2d_plot(name, v=view):
                            plot_2d_holder.clear()
                            with plot_2d_holder:
                                if name not in valid_runs:
                                    ui.label("Run not available.").classes("text-grey")
                                    return
                                _, projections = valid_runs[name]
                                if v not in projections:
                                    ui.label(f"View {v} not available.").classes("text-grey")
                                    return
                                slice_ = projections[v]
                                _plot(plot_slice(slice_, name=f"{name}: {v}"))
                                with ui.row():
                                    ui.button("PNG", on_click=lambda: _download_plot(slice_, ".png")).props("outline")
                                    ui.button("PDF", on_click=lambda: _download_plot(slice_, ".pdf")).props("outline")

                        run_selector.on_value_change(lambda e: update_2d_plot(e.value))
                        plot_2d_holder = ui.column().classes("w-full")
                        update_2d_plot(valid_names[0])

            view_tabs = ui.tabs(value=current_view,
                               on_change=lambda e: update_view(e.value)).classes("w-full")
            with view_tabs:
                for v in ordered_views:
                    ui.tab(v, label=v.replace("_", " ").title())

            plot_holder = ui.column().classes("w-full")
            update_view(current_view)

    holder_checkboxes = ui.column().classes("w-full")

    # Checkboxes for run selection
    ui.label("Select runs to include").classes("text-caption text-grey-7 mb-1")
    with holder_checkboxes:
        with ui.row().classes("w-full flex-wrap gap-2 mb-2"):
            for i, name in enumerate(all_names):
                checked = name in view_state["collimated_selected_runs"]
                cb = ui.checkbox(name, value=checked).props("dense")
                def on_toggle(event, n=name):
                    if event.value:
                        if n not in view_state["collimated_selected_runs"]:
                            view_state["collimated_selected_runs"].append(n)
                    else:
                        view_state["collimated_selected_runs"] = [x for x in view_state["collimated_selected_runs"] if x != n]
                    redraw(view_state["collimated_selected_runs"])
                cb.on_value_change(on_toggle)

    redraw(view_state["collimated_selected_runs"])