import os, datetime, pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType

# --- Load API Key ---
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro-002",
    temperature=0.5,
    google_api_key=os.environ.get("GOOGLE_API_KEY", "")
)

# --- Ensure dataset is available ---
csv_path = "online_sales_dataset.csv"

if not os.path.exists(csv_path):
    print("⚠ Dataset not found locally. Fetching from GCS bucket 'e-comm-assist'...")
    os.system("gsutil cp gs://e-comm-assist/online_sales_dataset.csv .")

if not os.path.exists(csv_path):
    raise FileNotFoundError("❌ Could not find dataset locally or in bucket 'e-comm-assist'")

# --- Load dataset ---
orders_df = pd.read_csv(csv_path)
print(f"✅ Loaded dataset: '{csv_path}' with {len(orders_df)} rows")

# --- Tool 1: Get order status ---
def get_order_status(order_id: str):
    order = orders_df[orders_df["InvoiceNo"].astype(str) == order_id.strip()]
    if order.empty:
        return f"❌ No order found for InvoiceNo {order_id}"
    row = order.iloc[0]
    return (
        f"📦 Order {row['InvoiceNo']} ({row['Description']})\n"
        f"- Quantity: {row['Quantity']}\n"
        f"- Unit Price: {row['UnitPrice']}\n"
        f"- Discount: {row['Discount']}\n"
        f"- Payment: {row['PaymentMethod']}\n"
        f"- Shipping: {row['ShippingCost']} via {row['ShipmentProvider']}\n"
        f"- Country: {row['Country']}\n"
        f"- Status: {row['ReturnStatus']}\n"
        f"- Priority: {row['OrderPriority']}\n"
        f"- Date: {row['InvoiceDate']}"
    )

# --- Tool 2: Initiate a return ---
def return_item(order_id: str):
    order = orders_df[orders_df["InvoiceNo"].astype(str) == order_id.strip()]
    if order.empty:
        return f"❌ Invalid InvoiceNo {order_id}. Cannot process return."
    row = order.iloc[0]
    return (
        f"✅ Return initiated for Order {order_id} ({row['Description']}).\n"
        f"A return label has been generated: https://dummylabel.com/{order_id}"
    )

# --- Tool 3: Get return policy ---
def get_return_policy(_: str = ""):
    return (
        "📜 Return Policy:\n"
        "- Items can be returned within 7 days of delivery.\n"
        "- Must be unused & in original packaging.\n"
        "- Refund within 5 business days after inspection."
    )

# --- Tool 4: Get product details ---
def get_product_details(product_name: str):
    product = orders_df[orders_df["Description"].str.contains(product_name, case=False, na=False)]
    if product.empty:
        return f"🔎 No product found for '{product_name}'"
    row = product.iloc[0]
    return (
        f"🛍 Product: {row['Description']}\n"
        f"- StockCode: {row['StockCode']}\n"
        f"- Category: {row['Category']}\n"
        f"- Unit Price: {row['UnitPrice']}\n"
        f"- Discount: {row['Discount']}\n"
        f"- Available via: {row['SalesChannel']}\n"
        f"- Shipped by: {row['ShipmentProvider']} from {row['WarehouseLocation']}"
    )

# --- Register tools ---
tools = [
    Tool(name="OrderStatusTool", func=get_order_status, description="Check order status by InvoiceNo"),
    Tool(name="ReturnTool", func=return_item, description="Initiate a return using InvoiceNo"),
    Tool(name="ReturnPolicyTool", func=get_return_policy, description="Get the company's return policy"),
    Tool(name="ProductInquiryTool", func=get_product_details, description="Get details about a product by name")
]

# --- Initialize Agent ---
agent = initialize_agent(
    tools, llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={
        "prefix": f"""
        You are E-Comm Assist, a friendly AI chatbot for an e-commerce store.
        Assist customers with orders, returns, and product questions.
        Always answer using the dataset provided.
        Today's date is {datetime.date.today().strftime('%B %d, %Y')}.
        """
    }
)

# --- Run Agent ---
def run_agent(query: str):
    return agent.run(query)
