"""
Config UI for Framework
--------------------------------
Interactive Tkinter-based configuration editor for all JSON config files.
Optimized for performance with async loading and Treeview-based editing.
"""

import json
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime
import logging
import shutil
import re
import threading
import time
from functools import partial

EXCLUDE_DIRS = {
    ".git", "__pycache__", "cache", "results", "logs", "reports", ".venv", "venv", "__tests__", "temp",
    ".audit_cache", ".pytest_cache", ".ruff_cache", ".test_cache", ".vscode", "htmlcov", "acceptance_reports",
    "api", "auditor", "backtest", "core", "data", "demo", "deploy", "docs", "etc", "examples", "experiments",
    "historical_data", "knowledge_base", "matrices", "md", "ml", "models", "monitoring", "my_cache",
    "nested", "notifier", "optimization", "performance_reports", "portfolio", "predictive_models",
    "README", "reporting", "risk", "safe_cache_dir", "scheduler", "scripts", "strategies", "templates",
    "test_historical_data", "test_logging_cache", "test_logs", "test_monitoring", "test_output",
    "test_refactor_cache", "tests", "tools", "utils", "valid_cache"
}

class ConfigUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Framework Config Manager")
        self.root.geometry("1200x800")
        self.config_files = []
        self.current_data = {}
        self.tabs = {}
        self.last_save_times = {}
        self.search_var = tk.StringVar()
        self.search_after_id = None
        self.loading_threads = []

        # Compiled regex for performance
        self.config_pattern = re.compile(r'(?:config|setting|param)', re.IGNORECASE)

        # Setup logging
        self._setup_logging()

        # Setup UI first
        self._setup_ui()

        # Start async config detection
        self._start_async_config_detection()

    def _setup_logging(self):
        """Setup logging for config changes."""
        log_path = Path("logs/config_edit.log")
        log_path.parent.mkdir(exist_ok=True)
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def _detect_configs_optimized(self):
        """Detect configuration JSONs using compiled regex for better performance."""
        start_time = time.time()
        configs = []
        root_dir = Path.cwd()

        for file in root_dir.glob("*.json"):
            if file.is_file() and self.config_pattern.search(file.name):
                configs.append(file)

        detection_time = time.time() - start_time
        logging.info(f"Config detection completed in {detection_time:.3f}s - found {len(configs)} files")
        return sorted(configs)

    def _start_async_config_detection(self):
        """Start asynchronous config file detection."""
        self.status_var.set("Loading config files...")
        self.root.update_idletasks()

        thread = threading.Thread(target=self._async_detect_configs, daemon=True)
        thread.start()
        self.loading_threads.append(thread)

    def _async_detect_configs(self):
        """Asynchronously detect config files and update UI."""
        try:
            configs = self._detect_configs_optimized()
            self.root.after(0, self._on_configs_detected, configs)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to detect config files: {e}"))

    def _on_configs_detected(self, configs):
        """Callback when config detection is complete."""
        self.config_files = configs

        # Print detected configs
        print("Detected config files:")
        for cfg in self.config_files:
            print(f"- {cfg.name}")

        # Update UI with detected configs
        self._setup_config_tabs()

        # Start loading config data asynchronously
        for cfg_file in self.config_files:
            thread = threading.Thread(target=self._async_load_config, args=(cfg_file,), daemon=True)
            thread.start()
            self.loading_threads.append(thread)

    def _setup_ui(self):
        """Setup the main UI components."""
        # Main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)

        # Search bar
        search_frame = ttk.Frame(self.main_frame)
        search_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(search_frame, text="Search:").pack(side="left")
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side="left", fill="x", expand=True, padx=(5, 0))

        # Bind search variable
        self.search_var.trace("w", self._on_search_change)

        # Progress bar for loading
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 5))

        # Notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)

        # Button frame
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill="x", padx=10, pady=5)

        self.save_button = ttk.Button(self.button_frame, text="Save", command=self._save_current, state="disabled")
        self.save_button.pack(side="left", padx=(0, 5))
        self.save_as_button = ttk.Button(self.button_frame, text="Save As...", command=self._save_as_current, state="disabled")
        self.save_as_button.pack(side="left", padx=(0, 5))
        self.reload_button = ttk.Button(self.button_frame, text="Reload", command=self._reload_current, state="disabled")
        self.reload_button.pack(side="left", padx=(0, 5))

        # Status bar
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief="sunken", anchor="w")
        self.status_bar.pack(fill="x", padx=10, pady=(0, 10))

    def _setup_config_tabs(self):
        """Setup tabs for detected config files."""
        if not self.config_files:
            label = ttk.Label(self.notebook, text="No configuration files detected.")
            self.notebook.add(label, text="No Configs")
            return

        for cfg_file in self.config_files:
            self._create_lazy_config_tab(cfg_file)

    def _create_lazy_config_tab(self, cfg_file):
        """Create a tab with lazy loading for a config file."""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text=cfg_file.name)
        self.tabs[cfg_file] = tab_frame

        # Loading indicator
        loading_label = ttk.Label(tab_frame, text="Loading configuration...")
        loading_label.pack(expand=True)

        # Store loading state
        tab_frame.loading_label = loading_label
        tab_frame.is_loaded = False

    def _async_load_config(self, cfg_file):
        """Asynchronously load a single config file."""
        try:
            start_time = time.time()
            with open(cfg_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            load_time = time.time() - start_time
            logging.info(f"Loaded {cfg_file.name} in {load_time:.3f}s")

            self.root.after(0, self._on_config_loaded, cfg_file, data)
        except Exception as e:
            self.root.after(0, self._on_config_load_error, cfg_file, str(e))

    def _on_config_loaded(self, cfg_file, data):
        """Callback when a config file is loaded."""
        self.current_data[cfg_file] = data.copy()

        # Mark as loaded
        tab_frame = self.tabs[cfg_file]
        tab_frame.is_loaded = True

        # Update progress
        loaded_count = sum(1 for tab in self.tabs.values() if getattr(tab, 'is_loaded', False))
        progress = (loaded_count / len(self.config_files)) * 100
        self.progress_var.set(progress)

        if loaded_count == len(self.config_files):
            self.status_var.set("All configurations loaded")
            self.progress_bar.pack_forget()  # Hide progress bar when done
            # Enable buttons when all configs are loaded
            self.save_button.config(state="normal")
            self.save_as_button.config(state="normal")
            self.reload_button.config(state="normal")
        else:
            self.status_var.set(f"Loaded {loaded_count}/{len(self.config_files)} configurations")

        # Handle tab content loading based on position
        cfg_index = self.config_files.index(cfg_file)
        if cfg_index == 0:
            # First tab: load immediately and select it
            self._load_tab_content(tab_frame, cfg_file)
            self.notebook.select(0)
        else:
            # Other tabs: create placeholder
            self._create_placeholder_tab(tab_frame, cfg_file)

    def _on_config_load_error(self, cfg_file, error_msg):
        """Handle config file loading errors."""
        tab_frame = self.tabs[cfg_file]
        tab_frame.is_loaded = True  # Mark as processed even on error

        # Remove loading indicator
        if hasattr(tab_frame, 'loading_label'):
            tab_frame.loading_label.destroy()

        # Show error message in tab
        error_label = ttk.Label(tab_frame, text=f"Failed to load {cfg_file.name}:\n{error_msg}", foreground="red")
        error_label.pack(expand=True)

        # Update progress
        loaded_count = sum(1 for tab in self.tabs.values() if getattr(tab, 'is_loaded', False))
        progress = (loaded_count / len(self.config_files)) * 100
        self.progress_var.set(progress)

        if loaded_count == len(self.config_files):
            self.status_var.set("All configurations loaded")
            self.progress_bar.pack_forget()
            # Enable buttons when all configs are loaded
            self.save_button.config(state="normal")
            self.save_as_button.config(state="normal")
            self.reload_button.config(state="normal")
        else:
            self.status_var.set(f"Loaded {loaded_count}/{len(self.config_files)} configurations")

        logging.error(f"Failed to load {cfg_file.name}: {error_msg}")

    def _create_placeholder_tab(self, tab_frame, cfg_file):
        """Create a placeholder tab that loads content when clicked."""
        # Remove loading indicator
        if hasattr(tab_frame, 'loading_label'):
            tab_frame.loading_label.destroy()

        # Create clickable placeholder
        placeholder_frame = ttk.Frame(tab_frame)
        placeholder_frame.pack(expand=True, fill="both")

        ttk.Label(placeholder_frame, text=f"📁 {cfg_file.name}", font=("Arial", 12, "bold")).pack(pady=(20, 10))
        click_label = ttk.Label(placeholder_frame, text="Click to load configuration", foreground="blue", cursor="hand2")
        click_label.pack(pady=(0, 20))

        # Make it clickable
        def on_click(event):
            self._load_tab_content(tab_frame, cfg_file)

        click_label.bind("<Button-1>", on_click)
        placeholder_frame.bind("<Button-1>", on_click)

        # Store reference
        tab_frame.placeholder_frame = placeholder_frame

    def _load_tab_content(self, tab_frame, cfg_file):
        """Load the actual content for a tab."""
        # Remove placeholder if it exists
        if hasattr(tab_frame, 'placeholder_frame'):
            tab_frame.placeholder_frame.destroy()

        # Create the actual content
        self._create_treeview_config_tab(tab_frame, cfg_file)

    def _create_treeview_config_tab(self, tab_frame, cfg_file):
        """Create a tab with Treeview-based editor for a config file."""
        # Create paned window for editor and preview
        paned = ttk.PanedWindow(tab_frame, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=10, pady=10)

        # Editor frame with Treeview
        editor_frame = ttk.Frame(paned)
        paned.add(editor_frame, weight=2)

        # Create Treeview
        columns = ("value", "type")
        tree = ttk.Treeview(editor_frame, columns=columns, show="tree headings", height=20)
        tree.heading("#0", text="Key")
        tree.heading("value", text="Value")
        tree.heading("type", text="Type")
        tree.column("#0", width=250, minwidth=150)
        tree.column("value", width=300, minwidth=200)
        tree.column("type", width=100, minwidth=80)

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(editor_frame, orient="vertical", command=tree.yview)
        h_scrollbar = ttk.Scrollbar(editor_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Layout
        tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        editor_frame.grid_rowconfigure(0, weight=1)
        editor_frame.grid_columnconfigure(0, weight=1)

        # Preview frame
        preview_frame = ttk.Frame(paned)
        paned.add(preview_frame, weight=1)

        ttk.Label(preview_frame, text="JSON Preview:").pack(anchor="w", padx=5, pady=5)
        preview_text = scrolledtext.ScrolledText(preview_frame, wrap="none", width=40)
        preview_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Store references
        tab_frame.tree = tree
        tab_frame.preview_text = preview_text

        # Populate tree
        self._populate_tree(tree, self.current_data[cfg_file], cfg_file)

        # Bind events
        tree.bind("<Double-1>", lambda e: self._on_tree_double_click(tree, cfg_file))
        tree.bind("<<TreeviewSelect>>", lambda e: self._update_preview(cfg_file))

        # Update preview
        self._update_preview(cfg_file)

    def _populate_tree(self, tree, data, cfg_file, parent="", path=""):
        """Populate Treeview with configuration data."""
        for key, value in data.items():
            full_path = f"{path}.{key}" if path else key
            item_id = f"{parent}.{key}" if parent else key

            if isinstance(value, dict):
                # Nested object
                node = tree.insert(parent, "end", text=key, values=("", "object"), open=True, tags=(full_path,))
                self._populate_tree(tree, value, cfg_file, node, full_path)
            elif isinstance(value, list):
                # Array
                node = tree.insert(parent, "end", text=key, values=(json.dumps(value), "array"), tags=(full_path,))
            else:
                # Primitive value
                type_name = type(value).__name__
                display_value = str(value)
                if isinstance(value, str) and len(display_value) > 50:
                    display_value = display_value[:47] + "..."
                node = tree.insert(parent, "end", text=key, values=(display_value, type_name), tags=(full_path,))

    def _on_tree_double_click(self, tree, cfg_file):
        """Handle double-click on tree item for editing."""
        selection = tree.selection()
        if not selection:
            return

        item = selection[0]
        values = tree.item(item, "values")
        if not values or len(values) < 2:
            return

        current_value = values[0]
        value_type = values[1]
        item_tags = tree.item(item, "tags")
        if not item_tags:
            return

        full_path = item_tags[0]

        # Create edit dialog
        self._show_edit_dialog(tree, item, cfg_file, full_path, current_value, value_type)

    def _show_edit_dialog(self, tree, item, cfg_file, path, current_value, value_type):
        """Show a dialog for editing a value."""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit {path}")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text=f"Path: {path}").pack(pady=5)
        ttk.Label(dialog, text=f"Type: {value_type}").pack(pady=5)

        # Value entry
        value_var = tk.StringVar(value=current_value)
        entry = ttk.Entry(dialog, textvariable=value_var)
        entry.pack(fill="x", padx=20, pady=10)
        entry.focus()

        def on_save():
            new_value = value_var.get()
            try:
                # Validate and convert value
                if value_type == "int":
                    new_value = int(new_value)
                elif value_type == "float":
                    new_value = float(new_value)
                elif value_type == "bool":
                    new_value = new_value.lower() in ("true", "1", "yes", "on")
                elif value_type == "array":
                    new_value = json.loads(new_value)
                    if not isinstance(new_value, list):
                        raise ValueError("Must be a JSON array")

                # Update data
                self._set_nested_value(self.current_data[cfg_file], path, new_value)

                # Update tree display
                display_value = str(new_value)
                if isinstance(new_value, str) and len(display_value) > 50:
                    display_value = display_value[:47] + "..."
                tree.item(item, values=(display_value, value_type))

                # Update preview
                self._update_preview(cfg_file)

                # Log change
                self._log_change(cfg_file, path, new_value)

                dialog.destroy()

            except Exception as e:
                messagebox.showerror("Error", f"Invalid value: {e}", parent=dialog)

        def on_cancel():
            dialog.destroy()

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill="x", padx=20, pady=10)

        ttk.Button(button_frame, text="Save", command=on_save).pack(side="left", padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side="left")

        # Bind Enter key
        dialog.bind("<Return>", lambda e: on_save())
        dialog.bind("<Escape>", lambda e: on_cancel())

    def _set_nested_value(self, data, path, value):
        """Set a value in a nested dictionary using dot notation."""
        keys = path.split('.')
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def _update_preview(self, cfg_file):
        """Update the JSON preview for a config file."""
        if cfg_file in self.tabs:
            tab_frame = self.tabs[cfg_file]
            if hasattr(tab_frame, 'preview_text'):
                preview_text = tab_frame.preview_text
                preview_text.delete("1.0", tk.END)
                preview_text.insert(tk.END, json.dumps(self.current_data[cfg_file], indent=2))

    def _on_search_change(self, *args):
        """Handle search input changes with debouncing."""
        # Cancel previous delayed search
        if self.search_after_id:
            self.root.after_cancel(self.search_after_id)

        # Schedule new search after delay
        self.search_after_id = self.root.after(300, self._perform_search)

    def _perform_search(self):
        """Perform the actual search operation."""
        search_term = self.search_var.get().lower().strip()
        current_tab = self.notebook.select()

        if not current_tab:
            return

        tab_frame = self.notebook.nametowidget(current_tab)

        # Handle Treeview search
        if hasattr(tab_frame, 'tree'):
            self._filter_tree_items(tab_frame.tree, search_term)

        self.search_after_id = None

    def _filter_tree_items(self, tree, search_term):
        """Filter Treeview items based on search term."""
        def search_item(item, parent_visible=False):
            """Recursively search through tree items."""
            item_text = tree.item(item, "text").lower()
            item_tags = tree.item(item, "tags")
            full_path = item_tags[0].lower() if item_tags else ""

            # Check if this item matches
            matches = (search_term in item_text) or (search_term in full_path)

            # Check children
            children = tree.get_children(item)
            has_visible_children = False
            for child in children:
                if search_item(child, matches or parent_visible):
                    has_visible_children = True

            # Show/hide this item
            visible = matches or has_visible_children or parent_visible or not search_term
            if visible:
                tree.item(item, open=True)  # Expand if visible
            else:
                tree.item(item, open=False)

            return visible

        # Start search from root
        for item in tree.get_children():
            search_item(item)

    def _save_current(self):
        """Save the currently active config file."""
        current_tab = self.notebook.select()
        if not current_tab:
            return

        tab_name = self.notebook.tab(current_tab, "text")
        cfg_file = None
        for file in self.config_files:
            if file.name == tab_name:
                cfg_file = file
                break

        if cfg_file:
            self._save_config_file(cfg_file)

    def _save_as_current(self):
        """Save current config as a new file."""
        current_tab = self.notebook.select()
        if not current_tab:
            return

        tab_name = self.notebook.tab(current_tab, "text")
        cfg_file = None
        for file in self.config_files:
            if file.name == tab_name:
                cfg_file = file
                break

        if cfg_file:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if file_path:
                new_file = Path(file_path)
                try:
                    with open(new_file, "w", encoding="utf-8") as f:
                        json.dump(self.current_data[cfg_file], f, indent=2)
                    self.last_save_times[new_file] = datetime.now()
                    self._log_change(new_file, "file", f"Saved as {new_file.name}")
                    self._update_status()
                    messagebox.showinfo("Success", f"Configuration saved as {new_file.name}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save: {e}")

    def _reload_current(self):
        """Reload the currently active config file."""
        current_tab = self.notebook.select()
        if not current_tab:
            return

        tab_name = self.notebook.tab(current_tab, "text")
        cfg_file = None
        for file in self.config_files:
            if file.name == tab_name:
                cfg_file = file
                break

        if cfg_file and messagebox.askyesno("Confirm", f"Reload {cfg_file.name} from disk? Unsaved changes will be lost."):
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.current_data[cfg_file] = data.copy()

                # Rebuild the tab
                tab_frame = self.tabs[cfg_file]
                for widget in tab_frame.winfo_children():
                    widget.destroy()

                self._create_treeview_config_tab(cfg_file)
                self._update_status()
                messagebox.showinfo("Success", f"{cfg_file.name} reloaded from disk")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reload: {e}")

    def _save_config_file(self, cfg_file):
        """Save a config file with atomic write and backup."""
        try:
            # Create backup
            backup_path = cfg_file.with_suffix('.bak')
            if cfg_file.exists():
                shutil.copy2(cfg_file, backup_path)

            # Atomic write
            temp_path = cfg_file.with_suffix('.tmp')
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.current_data[cfg_file], f, indent=2)

            # Move temp to final
            temp_path.replace(cfg_file)

            self.last_save_times[cfg_file] = datetime.now()
            self._log_change(cfg_file, "file", "Saved")
            self._update_status()
            messagebox.showinfo("Success", f"{cfg_file.name} saved successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save {cfg_file.name}: {e}")

    def _log_change(self, cfg_file, key, value):
        """Log a configuration change."""
        logging.info(f"{cfg_file.name} | {key} = {value}")

    def _update_status(self):
        """Update the status bar."""
        current_tab = self.notebook.select()
        if current_tab:
            tab_name = self.notebook.tab(current_tab, "text")
            cfg_file = None
            for file in self.config_files:
                if file.name == tab_name:
                    cfg_file = file
                    break

            if cfg_file:
                last_save = self.last_save_times.get(cfg_file, "Never")
                if isinstance(last_save, datetime):
                    last_save = last_save.strftime("%Y-%m-%d %H:%M:%S")
                self.status_var.set(f"Current file: {cfg_file.name} | Last saved: {last_save}")
            else:
                self.status_var.set("No file selected")
        else:
            self.status_var.set("No configuration files loaded")


if __name__ == "__main__":
    root = tk.Tk()
    app = ConfigUI(root)
    root.mainloop()
