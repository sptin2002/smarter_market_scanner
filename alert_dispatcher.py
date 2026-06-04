import os
import requests
from typing import List

class TelegramAlertDispatcher:
    """
    Decoupled Transmission Layer for broadcasting quantitative system
    reports to one or more Telegram user or group chat destinations.
    """
    def __init__(self):
        self.bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        raw_chat_ids: str = os.getenv("TELEGRAM_CHAT_ID", "")
        
        # Plain English: Split the comma-separated string into a clean list of individual IDs
        self.chat_ids: List[str] = [
            cid.strip() for cid in raw_chat_ids.split(",") if cid.strip()
        ]

    def broadcast_document(self, file_path: str, caption_text: str) -> None:
        """
        Loops through all configured Chat IDs and uploads the requested document file.
        """
        if not self.bot_token:
            print("[!] Skipping Telegram alert: TELEGRAM_BOT_TOKEN is missing from your environmental variable context.")
            return
            
        if not self.chat_ids:
            print("[!] Skipping Telegram alert: No valid destinations found inside TELEGRAM_CHAT_ID.")
            return

        telegram_url = f"https://api.telegram.org/bot{self.bot_token}/sendDocument"

        # Broadcast the file to every chat ID in your list
        for chat_id in self.chat_ids:
            try:
                print(f"[*] Uploading scan report to Telegram Target ID: {chat_id}...")
                
                # Re-open the file per transmission to avoid cursor position issues during the loop
                with open(file_path, "rb") as report_file:
                    payload_data = {"chat_id": chat_id, "caption": caption_text}
                    file_data = {"document": report_file}
                    
                    response = requests.post(
                        telegram_url, 
                        data=payload_data, 
                        files=file_data, 
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        print(f"[✓] Notification successfully delivered to target chat: {chat_id}")
                    else:
                        print(f"[❌] Telegram API rejected file for chat {chat_id}. Code {response.status_code}: {response.text}")
                        
            except Exception as network_error:
                print(f"[!] Critical network error connecting to Telegram for target {chat_id}: {network_error}")