import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests


class TokenFlowAnalyzer:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        # Add token prices in USD
        self.tokens = [
            ("CRV", "0x331b9182088e2a7d6d3fe4742aba1fb231aecc56", 1.0),
        ]
        self.token_prices = {token[0]: token[2] for token in self.tokens}

    @property
    def tokens(self):
        return self._tokens

    @tokens.setter
    def tokens(self, token_list):
        self._tokens = token_list
        # Update token prices whenever tokens are set
        self.token_prices = {token[0]: token[2] for token in token_list}

    def fetch_token_transfers(
        self, address: str, token_address: str
    ) -> List[Dict[str, Any]]:
        """Fetch all token transfers for a specific address and token."""
        all_transfers = []
        page = 1
        offset = 1000

        while True:
            params = {
                "module": "account",
                "action": "tokentx",
                "address": address,
                "contractaddress": token_address,
                "page": page,
                "offset": offset,
                "sort": "asc",
                "apikey": self.api_key,
            }

            try:
                response = requests.get(self.api_url, params=params)
                data = response.json()

                if data["status"] != "1" or not data["result"]:
                    break

                transfers = data["result"]
                all_transfers.extend(transfers)

                if len(transfers) < offset:
                    break

                page += 1
                time.sleep(0.2)  # Rate limiting

            except Exception as e:
                print(f"Error fetching data: {e}")
                break

        return all_transfers

    def process_transfers(
        self, transfers: List[Dict[str, Any]], address: str, token_symbol: str
    ) -> pd.DataFrame:
        """Process transfers into a DataFrame with calculated metrics including USD values."""
        records = []
        token_price = self.token_prices[token_symbol]

        for tx in transfers:
            decimals = int(tx.get("tokenDecimal", 18))
            amount = float(tx["value"]) / (10**decimals)
            is_inflow = tx["to"].lower() == address.lower()

            record = {
                "timestamp": datetime.fromtimestamp(int(tx["timeStamp"])),
                "hash": tx["hash"],
                "from": tx["from"],
                "to": tx["to"],
                "amount": amount,
                "usd_value": amount * token_price,
                "token": token_symbol,
                "flow_type": "inflow" if is_inflow else "outflow",
                "counterparty": tx["from"] if is_inflow else tx["to"],
            }
            records.append(record)

        return pd.DataFrame(records)

    def generate_flow_summary(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate a summary of token flows with USD values."""
        # Token amount summary
        amount_summary = (
            df.groupby(["token", "flow_type"])["amount"]
            .agg(["count", "sum", "mean", "median", "std"])
            .round(2)
        )

        # USD value summary
        usd_summary = (
            df.groupby(["token", "flow_type"])["usd_value"]
            .agg(["sum", "mean", "median", "std"])
            .round(2)
        )
        usd_summary.columns = ["usd_sum", "usd_mean", "usd_median", "usd_std"]

        # Combine summaries
        summary = pd.concat([amount_summary, usd_summary], axis=1)

        # Calculate net flows in both token and USD terms
        token_inflows = df[df["flow_type"] == "inflow"].groupby("token")["amount"].sum()
        token_outflows = (
            df[df["flow_type"] == "outflow"].groupby("token")["amount"].sum()
        )
        net_flows_token = (
            token_inflows - token_outflows.reindex(token_inflows.index).fillna(0)
        ).round(2)

        usd_inflows = (
            df[df["flow_type"] == "inflow"].groupby("token")["usd_value"].sum()
        )
        usd_outflows = (
            df[df["flow_type"] == "outflow"].groupby("token")["usd_value"].sum()
        )
        net_flows_usd = (
            usd_inflows - usd_outflows.reindex(usd_inflows.index).fillna(0)
        ).round(2)

        net_flows = pd.DataFrame(
            {"Net Flow (Token)": net_flows_token, "Net Flow (USD)": net_flows_usd}
        )

        return summary, net_flows

    def analyze_flows(self, address: str) -> pd.DataFrame:
        """Analyze flows for all tokens."""
        all_flows = pd.DataFrame()

        for (
            token_symbol,
            token_address,
            _,
        ) in self.tokens:  # Now correctly unpacking 3 values
            print(f"Fetching {token_symbol} transfers...")
            transfers = self.fetch_token_transfers(address, token_address)
            if transfers:
                df = self.process_transfers(transfers, address, token_symbol)
                all_flows = pd.concat([all_flows, df])

        return all_flows

    def generate_sankey_diagram(
        self, df: pd.DataFrame, min_value_usd: float = 100
    ) -> go.Figure:
        """Generate a Sankey diagram of token flows using USD values."""
        nodes = set()
        flows = defaultdict(float)

        # Get token colors
        token_colors = {
            token[0]: px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]
            for i, token in enumerate(self.tokens)
        }

        def shorten_address(addr: str) -> str:
            return addr[:6] + "..." + addr[-4:] if len(addr) > 12 else addr

        # Aggregate flows using USD values
        for token in df["token"].unique():
            token_df = df[df["token"] == token]

            for _, row in token_df.iterrows():
                if abs(row["usd_value"]) < min_value_usd:
                    continue

                if row["flow_type"] == "inflow":
                    source = f"{token}_{shorten_address(row['from'])}"
                    target = f"{token}_USER"
                else:
                    source = f"{token}_USER"
                    target = f"{token}_{shorten_address(row['to'])}"

                nodes.add(source)
                nodes.add(target)
                flows[(source, target)] += row["usd_value"]

        node_labels = list(nodes)
        node_dict = {node: idx for idx, node in enumerate(node_labels)}

        def format_node_label(label: str) -> str:
            token, addr = label.split("_")
            if addr == "USER":
                return f"{token} USER"
            return f"{token} {addr}"

        sankey_data = {
            "node": {
                "label": [format_node_label(label) for label in node_labels],
                "color": [token_colors[label.split("_")[0]] for label in node_labels],
            },
            "link": {
                "source": [node_dict[source] for source, _ in flows.keys()],
                "target": [node_dict[target] for _, target in flows.keys()],
                "value": list(flows.values()),
                "color": [
                    token_colors[source.split("_")[0]] for source, _ in flows.keys()
                ],
            },
        }

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=sankey_data["node"],
                    link=sankey_data["link"],
                    arrangement="snap",
                    valueformat="$,.0f",
                )
            ]
        )

        fig.update_layout(
            title={
                "text": "Token Flow Analysis (USD Values)",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            font_size=10,
            height=800,
            margin=dict(t=100, l=50, r=50, b=50),
        )

        return fig
