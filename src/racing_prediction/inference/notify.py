"""予測結果の通知 (メール / ntfy).

設定はすべて環境変数から読む (GitHub Actions では Secrets に登録する).

メール (SMTP):
    SMTP_HOST      例: smtp.gmail.com
    SMTP_PORT      例: 587 (STARTTLS) / 465 (SSL). 省略時 587
    SMTP_USER      送信元アカウント
    SMTP_PASSWORD  パスワード (Gmail はアプリパスワード)
    MAIL_TO        送信先 (カンマ区切りで複数可). 省略時は SMTP_USER
ntfy (スマホのプッシュ通知アプリ. 任意):
    NTFY_TOPIC     購読するトピック名 (推測されにくい長い文字列にする)
    NTFY_SERVER    省略時 https://ntfy.sh
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.mime.text import MIMEText

import requests


def email_configured() -> bool:
    return all(os.environ.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"))


def ntfy_configured() -> bool:
    return bool(os.environ.get("NTFY_TOPIC"))


def send_email(subject: str, body: str) -> None:
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ["SMTP_USER"]
    password = os.environ["SMTP_PASSWORD"]
    to = [a.strip() for a in os.environ.get("MAIL_TO", user).split(",") if a.strip()]

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = ", ".join(to)

    context = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as smtp:
            smtp.login(user, password)
            smtp.sendmail(user, to, msg.as_string())
    else:
        with smtplib.SMTP(host, port, timeout=30) as smtp:
            smtp.starttls(context=context)
            smtp.login(user, password)
            smtp.sendmail(user, to, msg.as_string())


def send_ntfy(title: str, body: str) -> None:
    # 日本語のタイトルをヘッダーで送ると文字化けするため JSON で送る
    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    response = requests.post(
        server,
        json={"topic": os.environ["NTFY_TOPIC"], "title": title, "message": body, "tags": ["horse_racing"]},
        timeout=30,
    )
    response.raise_for_status()


def notify(subject: str, body: str) -> list[str]:
    """設定されている全ての通知先に送る. 送った通知先の名前のリストを返す.

    1 つも設定されていなければ RuntimeError. 一部の送信に失敗した場合も, 全部送り終えてから例外を出す.
    """
    senders = []
    if email_configured():
        senders.append(("email", send_email))
    if ntfy_configured():
        senders.append(("ntfy", send_ntfy))
    if not senders:
        raise RuntimeError("通知先が設定されていません (SMTP_* または NTFY_TOPIC の環境変数を設定してください)")

    sent, errors = [], []
    for name, send in senders:
        try:
            send(subject, body)
            sent.append(name)
        except Exception as e:  # 片方が失敗してももう片方は送る
            errors.append(f"{name}: {e}")
    if errors and not sent:
        raise RuntimeError("通知の送信に失敗しました: " + " / ".join(errors))
    for err in errors:
        print("警告: 通知の一部に失敗しました:", err)
    return sent
