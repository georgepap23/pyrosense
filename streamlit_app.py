import streamlit as st
import folium
from streamlit_folium import st_folium
import datetime
import pandas as pd
import joblib
import time
import subprocess
import tempfile
from pathlib import Path

# Set up the page configuration
st.set_page_config(page_title="PyroSense Dashboard", layout="wide", page_icon="🔥")

# --- 1. INITIALIZE GLOBAL SESSION STATE ---
if "lat" not in st.session_state:
    st.session_state.lat = 37.9838
if "lon" not in st.session_state:
    st.session_state.lon = 23.7275
if "date" not in st.session_state:
    st.session_state.date = datetime.date.today()
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. SIDEBAR: GLOBAL SETTINGS & LOGO ---
# Check if logo exists, otherwise skip
logo_path = Path("assets/pyrosense.png")
if logo_path.exists():
    st.sidebar.image(str(logo_path), use_column_width=True)
else:
    st.sidebar.title("🔥 PyroSense")
st.sidebar.markdown("---")

st.sidebar.subheader("📅 Global Context")
# Update global date
st.session_state.date = st.sidebar.date_input("Analysis Date", st.session_state.date)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Coordinate Entry")

# Manual Inputs - Synced with Session State
# Note: key parameters ensure Streamlit tracks these specific widgets
input_lat = st.sidebar.number_input("Latitude", value=st.session_state.lat, format="%.6f", step=0.01)
input_lon = st.sidebar.number_input("Longitude", value=st.session_state.lon, format="%.6f", step=0.01)

# Sync logic: If manual input changes, update session state
if input_lat != st.session_state.lat or input_lon != st.session_state.lon:
    st.session_state.lat = input_lat
    st.session_state.lon = input_lon

st.sidebar.markdown("---")
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.info(f"📍 **Target Locked:**\n{st.session_state.lat:.4f}, {st.session_state.lon:.4f}")

# ==========================================
# 3. PERSISTENT MAP SECTION (TOP)
# ==========================================
st.header("📍 Location Intelligence")
st.write("Click the map to update coordinates or refine via the sidebar.")

# Create map centered on current session state
m = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=6)
folium.Marker(
    [st.session_state.lat, st.session_state.lon], 
    popup="Current Target", 
    icon=folium.Icon(color='red', icon='fire')
).add_to(m)
m.add_child(folium.LatLngPopup())

# Render map and capture click data
map_data = st_folium(m, height=400, use_container_width=True, key="main_map")

# Sync map click back to session state
if map_data and map_data.get("last_clicked"):
    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]
    if clicked_lat != st.session_state.lat or clicked_lon != st.session_state.lon:
        st.session_state.lat = clicked_lat
        st.session_state.lon = clicked_lon
        st.rerun()  # Rerun to update sidebar and map marker

st.divider()

# ==========================================
# 4. ANALYSIS TABS (BOTTOM)
# ==========================================
tab1, tab2 = st.tabs(["📊 Stacking Risk Model", "🌍 EarthDial VLM Analysis"])

# --- TAB 1: STACKING RISK MODEL ---
with tab1:
    st.subheader("Predictive Risk Assessment")
    st.write(f"Evaluating risk for **{st.session_state.date}** at **{st.session_state.lat:.4f}, {st.session_state.lon:.4f}**")
    
    model_path = st.text_input("Model Path", value="data/stacking_ensemble.pkl")
    
    if st.button("Calculate Probability", type="primary"):
        with st.status("Extracting features...") as s:
            st.write("🛰️ Downloading HLS satellite imagery...")
            st.write("🌤️ Fetching weather history...")
            st.write("🤖 Running ML prediction...")

            try:
                # Call pyrosense predict CLI (use venv if available)
                venv_pyrosense = Path("venv/bin/pyrosense")
                pyrosense_cmd = str(venv_pyrosense) if venv_pyrosense.exists() else "pyrosense"

                cmd = [
                    pyrosense_cmd, "predict",
                    "--model", model_path,
                    "--lat", str(st.session_state.lat),
                    "--lon", str(st.session_state.lon),
                    "--date", st.session_state.date.strftime("%Y-%m-%d"),
                    "--prithvi-model", "Prithvi-EO-2.0-300M"
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    # Parse output to extract fire probability
                    output = result.stdout
                    for line in output.split('\n'):
                        if 'Fire Probability:' in line:
                            fire_prob_str = line.split(':')[1].strip().rstrip('%')
                            fire_prob = float(fire_prob_str) / 100
                            break
                    else:
                        fire_prob = 0.5  # Default if parsing fails

                    s.update(label="Analysis Complete", state="complete")
                else:
                    # Check if error is due to missing HLS imagery
                    error_msg = result.stderr
                    if "No HLS imagery found" in result.stdout or "No valid imagery found" in error_msg:
                        st.warning("⚠️ **No satellite imagery available**")
                        st.info("""
                        No cloud-free HLS satellite imagery was found for this location and date.

                        **Possible reasons:**
                        - Heavy cloud cover during the search period
                        - Location is over ocean or outside coverage area
                        - Imagery not yet available for recent dates

                        **Suggestions:**
                        - Try a different date (earlier or later)
                        - Try a nearby location
                        - Use the EarthDial VLM Analysis tab instead (can analyze any available imagery)
                        """)
                    else:
                        st.error(f"Prediction failed: {error_msg}")
                    s.update(label="No Imagery Available", state="error")
                    fire_prob = None

            except subprocess.TimeoutExpired:
                st.error("Prediction timed out (>5 minutes)")
                s.update(label="Timeout", state="error")
                fire_prob = None
            except Exception as e:
                st.error(f"Error: {e}")
                s.update(label="Error", state="error")
                fire_prob = None

        if fire_prob is not None:
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Fire Probability", f"{fire_prob:.1%}")

            if fire_prob > 0.7:
                c2.error(f"🔴 RISK LEVEL: HIGH")
            elif fire_prob > 0.4:
                c2.warning(f"🟡 RISK LEVEL: MEDIUM")
            else:
                c2.success(f"🟢 RISK LEVEL: LOW")

# --- TAB 2: EARTHDIAL VLM ---
with tab2:
    st.subheader("EarthDial Satellite Intelligence")
    st.write("Automated HLS download and visual analysis.")

    with st.expander("⚙️ Model Hardware Settings", expanded=False):
        device = st.selectbox("Device", ["auto", "cuda", "mps", "cpu"], index=0)
        load_8bit = st.checkbox("Use 8-bit quantization", value=True)

    col_report, col_chat = st.columns([1, 1.2])

    with col_report:
        if st.button("Generate Intelligence Report", type="primary"):
            with st.status("Fetching Satellite Imagery...") as status:
                st.write(f"Downloading HLS imagery for {st.session_state.date}...")
                st.write("Loading EarthDial VLM...")
                st.write("Running visual analysis...")

                try:
                    # Call pyrosense analyze CLI (use venv if available)
                    venv_pyrosense = Path("venv/bin/pyrosense")
                    pyrosense_cmd = str(venv_pyrosense) if venv_pyrosense.exists() else "pyrosense"

                    cmd = [
                        pyrosense_cmd, "analyze",
                        "--lat", str(st.session_state.lat),
                        "--lon", str(st.session_state.lon),
                        "--date", st.session_state.date.strftime("%Y-%m-%d"),
                        "--device", device,
                    ]
                    if load_8bit:
                        cmd.append("--load-8bit")

                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

                    if result.returncode == 0:
                        output = result.stdout
                        status.update(label="Analysis Finished!", state="complete")

                        # Parse the structured report from CLI output
                        report_sections = {}
                        current_section = None
                        fire_probability = None

                        for line in output.split('\n'):
                            # Extract fire probability if present
                            if 'Fire Probability:' in line:
                                try:
                                    fire_prob_str = line.split(':')[1].strip().rstrip('%')
                                    fire_probability = float(fire_prob_str)
                                except:
                                    pass

                            # Parse sections
                            if 'SUMMARY:' in line:
                                current_section = 'summary'
                                report_sections[current_section] = []
                            elif 'VEGETATION:' in line:
                                current_section = 'vegetation'
                                report_sections[current_section] = []
                            elif 'TERRAIN:' in line:
                                current_section = 'terrain'
                                report_sections[current_section] = []
                            elif 'STRATEGIES:' in line:
                                current_section = 'strategies'
                                report_sections[current_section] = []
                            elif 'RISK ASSESSMENT:' in line:
                                current_section = 'risk'
                                report_sections[current_section] = []
                            elif current_section and line.strip() and not line.startswith('=') and 'Fire Probability' not in line and 'Location:' not in line:
                                report_sections[current_section].append(line.strip())

                        # Display the report
                        if report_sections:
                            st.markdown("### 🔍 EARTHDIAL INTELLIGENCE REPORT")

                            if fire_probability is not None:
                                st.metric("🔥 Fire Probability", f"{fire_probability:.1f}%")

                            if report_sections.get('summary'):
                                st.markdown("**📊 Summary:**")
                                st.write('\n'.join(report_sections['summary']))

                            if report_sections.get('vegetation'):
                                st.markdown("**🌿 Vegetation Analysis:**")
                                st.write('\n'.join(report_sections['vegetation']))

                            if report_sections.get('terrain'):
                                st.markdown("**⛰️ Terrain Factors:**")
                                st.write('\n'.join(report_sections['terrain']))

                            if report_sections.get('strategies'):
                                st.markdown("**🚒 Recommended Strategies:**")
                                st.write('\n'.join(report_sections['strategies']))

                            if report_sections.get('risk'):
                                st.markdown("**⚠️ Risk Assessment:**")
                                st.write('\n'.join(report_sections['risk']))
                        else:
                            st.warning("Report generated but couldn't parse sections. Raw output:")
                            st.code(output)
                    else:
                        st.error(f"Analysis failed: {result.stderr}")
                        status.update(label="Analysis Failed", state="error")

                except subprocess.TimeoutExpired:
                    st.error("Analysis timed out (>10 minutes)")
                    status.update(label="Timeout", state="error")
                except Exception as e:
                    st.error(f"Error: {e}")
                    status.update(label="Error", state="error")

    with col_chat:
        st.write("💬 **Contextual Chat**")
        # Scrollable chat container
        chat_box = st.container(height=400)
        with chat_box:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about this terrain..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_box:
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        try:
                            # Call pyrosense analyze with custom question (use venv if available)
                            venv_pyrosense = Path("venv/bin/pyrosense")
                            pyrosense_cmd = str(venv_pyrosense) if venv_pyrosense.exists() else "pyrosense"

                            cmd = [
                                pyrosense_cmd, "analyze",
                                "--lat", str(st.session_state.lat),
                                "--lon", str(st.session_state.lon),
                                "--date", st.session_state.date.strftime("%Y-%m-%d"),
                                "--question", prompt,
                                "--device", device,
                            ]
                            if load_8bit:
                                cmd.append("--load-8bit")

                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                            if result.returncode == 0:
                                # Extract the response from CLI output
                                output = result.stdout
                                # The response is between "EarthDial Analysis:" and the final "==="
                                lines = output.split('\n')
                                response_lines = []
                                capture = False
                                found_header = False

                                for line in lines:
                                    if 'EarthDial Analysis:' in line:
                                        found_header = True
                                        continue
                                    if found_header and line.startswith('==='):
                                        # Skip the separator right after the header
                                        capture = True
                                        continue
                                    if capture and line.startswith('==='):
                                        # End of response
                                        break
                                    if capture and line.strip():
                                        response_lines.append(line.strip())

                                reply = '\n'.join(response_lines) if response_lines else "Analysis completed, but couldn't parse response."
                            else:
                                reply = f"Error analyzing image: {result.stderr[:200]}"

                        except subprocess.TimeoutExpired:
                            reply = "Analysis timed out. Try a simpler question."
                        except Exception as e:
                            reply = f"Error: {str(e)}"

                        st.markdown(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})