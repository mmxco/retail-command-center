# Workspace Coding Rules & Constraints

## Streamlit API Standards
- **Never use deprecated `use_container_width`**: The `use_container_width` argument was deprecated in Streamlit (scheduled for removal post-2025).
- **Full-Width Widgets**: Always use `width="stretch"` for elements that should span the container (e.g., `st.button("...", width="stretch")`, `st.link_button("...", width="stretch")`, `st.download_button("...", width="stretch")`).
- **DataFrames & Plotly Charts**: In modern Streamlit, `width="stretch"` is already the default for `st.dataframe` and `st.plotly_chart`. Omit `use_container_width` entirely or explicitly set `width="stretch"`.
