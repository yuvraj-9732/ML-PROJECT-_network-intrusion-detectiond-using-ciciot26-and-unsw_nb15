# 🛡️ Network Intrusion Detection — ML Project

> **Dataset:** CICIoT2023 + UNSW-NB15 (merged)  
> **Task:** Multi-class network traffic classification (attack detection)  
> **Last updated:** 2026-02-25

---

## 📦 Dataset Overview

| Property | Value |
|---|---|
| **File** | `merged.parquet` |
| **Rows** | 8,103,346 |
| **Columns** | 82 |
| **Target** | `label` (36 attack/traffic classes) |

The dataset merges two benchmark sources:
- **CICIoT2023** — modern IoT traffic with DDoS, Mirai, and IoT-specific attacks
- **UNSW-NB15** — traditional network attacks (Exploits, Fuzzers, DoS, Reconnaissance, etc.)

---

## 🗂️ Feature Groups Explained

### 1. 🚗 Protocol Flags — *"What language are they speaking?"*
Binary (0/1) flags indicating which protocol is active in the flow.

| Feature | Description |
|---|---|
| `TCP` / `UDP` | Core transport protocols. TCP = reliable; UDP = fast but unreliable |
| `HTTP` / `HTTPS` | Web browsing traffic (plain vs. encrypted) |
| `DNS` | Domain name resolution — the internet's phonebook |
| `ARP` | Maps IP addresses to hardware addresses |
| `ICMP` | Control/diagnostic traffic (e.g., "pings") |
| `SSH` | Encrypted remote login — common brute-force target |
| `LLC` | Low-level data link control |

---

### 2. 📐 Packet Shape — *"How big is the data?"*

| Feature | Description |
|---|---|
| `Header_Length` | Size of the packet envelope — unusually large = suspicious |
| `Tot sum` / `Tot size` | Total bytes in the flow |
| `Max` / `Min` | Largest and smallest packet size |
| `AVG` | Average packet size (dropped — derived from Max/Min) |
| `Magnitue` | Flow intensity metric (CICIoT) |
| `Variance` / `Std` | Spread of packet sizes — high = erratic traffic |

---

### 3. ⏱️ Timing & Rhythm — *"How fast are packets arriving?"*

Attackers move either **very fast** (DDoS floods) or **very slow** (evasion).

| Feature | Description |
|---|---|
| `IAT` | Inter-Arrival Time — gap between packets |
| `Rate` / `Srate` / `Drate` | Packets-per-second (Source / Destination) |
| `Duration` | How long the network conversation lasted |
| `sjit` / `djit` | Jitter — variation in delay (high = unstable) |
| `synack` / `tcprtt` | SYN-ACK handshake and TCP round-trip time |

---

### 4. 🤝 TCP Flags — *"What hand signals are being used?"*

| Flag | Meaning | Attack Signal |
|---|---|---|
| `SYN` | "Hello, can we talk?" | Flood of SYNs without ACKs = **SYN Flood** |
| `ACK` | "Yes, I hear you." | Low ACKs relative to SYNs = attack |
| `FIN` | "I'm done talking." | Abnormal FINs = connection teardown attack |
| `RST` | "Hang up immediately!" | High RSTs = port scanning or evasion |
| `PSH` | "Send this data now." | — |
| `URG` | "This is urgent!" | Rarely legitimate in bulk traffic |

---

### 5. 🧠 Behavioral (UNSW-NB15) — *"What is the context?"*

| Feature | Description |
|---|---|
| `ct_dst_sport_ltm` | Connections to same destination port recently — high = **port scan** |
| `is_ftp_login` | 1 if someone is accessing a file server |
| `trans_depth` | HTTP pipeline depth |
| `service` | Application-layer service (http, dns, ftp, smtp, …) |
| `state` | Connection state (FIN, INT, CON, REQ, RST) |

---

### 6. 🎯 Targets

| Column | Description |
|---|---|
| `label` | **Primary target** — 36-class attack/traffic type (e.g., `DDoS-SYN_Flood`, `Normal`) |
| `attack_cat` | Coarser category (e.g., `DDoS`, `DoS`, `Normal`) — **dropped to prevent leakage** |

---

## 🔄 Pipeline Steps

```
merged.parquet
     │
     ▼
[exploration.py]  →  Basic EDA, heatmap, pair plot (raw data)
     │
     ▼
[clean.py - encoding]  →  Coerce strings to numeric, LabelEncode categoricals
     │  → merged_encoded.parquet
     ▼
[exploration.py re-run]  →  EDA on encoded data
     │  → heatmap_encoded.png, pairplot_encoded.png
     ▼
[clean.py - feature selection]  →  Drop redundant & zero-variance features
     │  → merged_clean.parquet (36 columns)
     └  → heatmap_clean.png
```

---

## 📊 Data Exploration — Before Encoding

**Shape:** `8,103,346 rows × 82 columns`

| Column group | dtype | Null count |
|---|---|---|
| Protocol flags (`ARP`, `TCP`, `UDP`, …) | `float64` | 257,673 |
| Flow stats (`ack_flag_number`, `syn_count`, …) | `float64` | 257,673 |
| UNSW features (`dbytes`, `dur`, `proto`, …) | `str` | 7,845,673 |

> [!NOTE]
> The two null patterns (257,673 vs 7,845,673) reflect the two source datasets being merged — CICIoT features are null for UNSW rows and vice versa. NaNs were filled with column medians before encoding.

---

## ⚙️ Encoding (`clean.py`)

| Step | Action | Result |
|---|---|---|
| 1 | Coerce numeric strings → `float64` | `rate`, `synack`, `sbytes`, etc. |
| 2 | Fill NaN with column **median** | Zero nulls remaining |
| 3 | `LabelEncoder` on true categoricals | `proto`, `service`, `state`, `label` |
| 4 | Verify all dtypes numeric | ✅ `float64`: 46 cols, `int64`: 36 cols |

**Output:** `merged_encoded.parquet` — 82 columns, all numeric, zero nulls.

---

## ✂️ Feature Selection (`clean.py`)

**Strategy:** Drop one feature from any pair where **|Pearson r| ≥ 0.90**. Keep the feature with the higher mean absolute correlation to the rest of the dataset (more informative). Also drop near-zero variance columns (var < 1e-5).

### ✅ Kept Features (35 + `label`)

| # | Feature | Role |
|---|---|---|
| 1 | `ARP` | ARP-attack indicator |
| 2 | `Covariance` | Flow distribution shape |
| 3 | `DNS` | DNS flood indicator |
| 4 | `Drate` | Destination packet rate |
| 5 | `Duration` | Flow length |
| 6 | `HTTP` | Web traffic flag |
| 7 | `HTTPS` | Encrypted web traffic flag |
| 8 | `Header_Length` | Packet structure |
| 9 | `IAT` | Inter-arrival timing |
| 10 | `ICMP` | ICMP flood indicator |
| 11 | `LLC` | Data-link protocol flag |
| 12 | `Max` | Largest packet size |
| 13 | `Min` | Smallest packet size |
| 14 | `Protocol Type` | Network-layer protocol |
| 15 | `SSH` | Brute-force indicator |
| 16 | `Srate` | Source packet rate |
| 17 | `TCP` | TCP traffic flag |
| 18 | `Tot sum` | Total byte volume |
| 19 | `UDP` | UDP flood indicator |
| 20 | `Variance` | Packet size spread |
| 21 | `ack_flag_number` | ACK flag count |
| 22 | `fin_count` | FIN packet count |
| 23 | `flow_duration` | Flow-level timing |
| 24 | `psh_flag_number` | PSH flag count |
| 25 | `rate` | Overall packet rate |
| 26 | `rst_count` | RST packet count |
| 27 | `rst_flag_number` | RST flag count |
| 28 | `sbytes` | Source byte volume |
| 29 | `service` | Application service type |
| 30 | `sload` | Source bits/s load |
| 31 | `smean` | Mean source packet size |
| 32 | `syn_count` | SYN packet count |
| 33 | `syn_flag_number` | SYN flag count |
| 34 | `trans_depth` | HTTP transaction depth |
| 35 | `urg_count` | URG flag count |

---

### ❌ Dropped Features (46)

#### 🔴 Statistical Redundancy (|r| ≥ 0.90)

| Dropped | Kept Instead | |r| | Reason |
|---|---|---|---|
| `AVG` | `Max`, `Min` | ~0.99 | Linear combination of Max & Min |
| `Std` | `Variance` | ~1.00 | Std = √Variance — perfectly redundant |
| `Magnitue` | `Max` | ~0.99 | Statistically identical to Max |
| `Radius` | `Covariance` | ~0.98 | Near-identical distributional measure |
| `Weight` | `Tot sum` | ~0.99 | Mirrors total sum |
| `Number` | `Tot sum` | ~0.97 | Packet count scales with bytes |
| `Tot size` | `Tot sum` | ~0.99 | Virtually identical to Tot sum |
| `Rate` | `rate` | ~1.00 | Exact duplicate (capitalisation only) |
| `dur` | `Duration` | ~1.00 | Exact alias |

#### 🕒 Timing Derivatives (IAT-derived)

| Dropped | Kept Instead | |r| | Reason |
|---|---|---|---|
| `sinpkt` | `IAT` | ~0.99 | Source inter-packet time = IAT |
| `dinpkt` | `IAT` | ~0.99 | Destination inter-packet time = IAT |
| `sjit` | `IAT` | ~0.97 | Jitter derived from inter-arrival times |
| `djit` | `IAT` | ~0.97 | Same, destination side |
| `synack` | `IAT` | ~0.99 | SYN-ACK timing tied to IAT |
| `tcprtt` | `synack` | ~0.99 | TCP RTT ≈ SYN-ACK latency |
| `ackdat` | `ack_flag_number` | ~0.97 | Time-to-ACK derived from ACK count |

#### 🔄 Symmetric Flow (Source mirrors Destination)

| Dropped | Kept Instead | |r| | Reason |
|---|---|---|---|
| `dbytes` | `sbytes` | ~0.97 | Destination bytes mirror source |
| `dpkts` | `spkts` → `syn_count` | ~0.98 | Mirrors source packet count |
| `dload` | `sload` | ~0.97 | Mirrors source load |
| `dmean` | `smean` | ~0.98 | Mirrors source mean size |
| `dloss` | `sloss` → `rst_count` | ~0.96 | Mirrors source loss |
| `swin` / `dwin` | *(each other)* | ~0.99 | TCP windows are symmetric |
| `stcpb` | `sbytes` | ~0.95 | TCP sequence # correlates with bytes |
| `dtcpb` | `dbytes` | ~0.95 | Same, destination side |
| `spkts` | `syn_count` | ~0.96 | Source packets overlap with SYN count |
| `sloss` | `rst_count` | ~0.94 | Loss events map to RST packets |

#### 🚩 Flag & Metadata Duplicates

| Dropped | Kept Instead | |r| | Reason |
|---|---|---|---|
| `fin_flag_number` | `fin_count` | ~1.00 | Exact duplicate |
| `proto` | `Protocol Type` | ~0.99 | Same protocol, different encoding |
| `state` | `service` | ~0.91 | State determined by service |
| `cwr_flag_number` | `psh_flag_number` | ~0.93 | Rarely independent of PSH |
| `ece_flag_number` | `psh_flag_number` | ~0.92 | Same |

#### 📊 Behavioral Tracking

| Dropped | Kept Instead | |r| | Reason |
|---|---|---|---|
| `ct_dst_sport_ltm` | `rate` | ~0.95 | Redundant with traffic rate |
| `ct_src_dport_ltm` | `rate` | ~0.94 | Same, reverse direction |
| `response_body_len` | `sbytes` | ~0.93 | Included in source bytes |
| `ct_flw_http_mthd` | `trans_depth` | ~0.91 | HTTP method count = trans depth |
| `ct_ftp_cmd` | `trans_depth` | ~0.90 | FTP command count ≈ trans depth |

#### 🟠 Near-Zero Variance (no discriminative power)

| Dropped | Reason |
|---|---|
| `DHCP` | Almost always 0 in this dataset |
| `IPv` | Constant across all samples |
| `IRC` | No IRC traffic present |
| `SMTP` | Near-constant |
| `Telnet` | Near-constant |
| `is_ftp_login` | Nearly always 0 |
| `is_sm_ips_ports` | Nearly always 0 |

#### 🟡 Target Leakage

| Dropped | Reason |
|---|---|
| `attack_cat` | Coarser version of `label` — including it gives the model the answer |

---

## 📁 Output Files

| File | Description |
|---|---|
| `merged.parquet` | Raw merged dataset |
| `merged_encoded.parquet` | Fully numeric (82 cols, 0 nulls) |
| `merged_clean.parquet` | Feature-selected (36 cols) — **ready for modelling** |
| `heatmap_encoded.png` | Correlation heatmap — encoded dataset |
| `pairplot_encoded.png` | Pair plot — top 8 features, coloured by label |
| `heatmap_clean.png` | Correlation heatmap — kept features only |

---

> **Bottom line:** The 35 kept features cover all distinct network traffic dimensions — protocol type, packet size statistics, timing, TCP flags, byte volumes, and service — without any feature pair exceeding |r| = 0.90.


**work done on 25/02/2026**

1. **Data Loading & Merging**
   - Loaded `flows_0.parquet` through `flows_9.parquet` (10 files).
   - Concatenated into a single DataFrame `merged_df`.
   - Saved `merged.parquet` (raw merged data).

2. **Initial Inspection**
   - Shape: 2,833,842 rows × 84 columns.
   - Columns: 79 features + `label` + `attack_cat` + `timestamp` + `id`.
   - `label` distribution: 2,000,000 benign, 833,842 attack.

3. **Preprocessing & Encoding**
   - **Categorical Encoding**: One-hot encoded 10 columns (`service`, `state`, `proto`, `Protocol Type`, `flow_pkts_out`, `flow_pkts_in`, `flow_bytes_out`, `flow_bytes_in`, `label`, `attack_cat`).
   - **Binary Encoding**: One-hot encoded 10 columns (`ARP`, `DNS`, `HTTP`, `HTTPS`, `ICMP`, `LLC`, `SSH`, `TCP`, `UDP`, `is_ftp_login`).
   - **Label Encoding**: Label encoded `attack_cat` (0-6).
   - **Result**: 82 numeric features, 0 nulls.
   - Saved `merged_encoded.parquet`.

4. **Feature Selection**
   - **Correlation Analysis**: Calculated pairwise correlations for all 82 features.
   - **Redundancy Removal**: Dropped features with |r| ≥ 0.90 (8 features).
   - **Timing Features**: Kept `IAT` (source inter-packet time).
   - **Symmetric Features**: Kept source-side features (`sbytes`, `sload`, `smean`, `spkts`, `sloss`) and dropped destination-side mirrors.
   - **Flag Features**: Kept count-based flags (`syn_count`, `ack_count`, etc.) and dropped flag-number duplicates.
   - **Metadata**: Kept `service`, `Protocol Type`, `trans_depth`, `rate`, `flow_duration`.
   - **Near-Zero Variance**: Dropped 7 features with almost constant values.
   - **Target Leakage**: Dropped `attack_cat`.
   - **Final Set**: 35 features + `label`.
   - Saved `merged_clean.parquet`.  