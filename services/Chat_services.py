
# === Load .env ===
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

async def get_chat_history(user_id):
    try:
        res = supabase.table("conversations") \
            .select("question, answer") \
            .eq("user_id", user_id) \
            .order("last_updated", desc=False) \
            .execute()
        return {"history": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
