"""Run the Ledger web app on localhost."""

import argparse
import io
import json
import re
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from ledger import Classifier, LABELS, MAX_FILE, MAX_TEXT, ROOT, ROUTER_NAME

STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
MAX_PDF_PAGES = 40
MAX_DOCX_XML = 4 * 1024 * 1024


def extract_pdf(body):
    try:
        reader = PdfReader(io.BytesIO(body))
        if reader.is_encrypted:
            raise ValueError("Unlock the PDF before uploading it.")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise ValueError(f"Upload a PDF with at most {MAX_PDF_PAGES} pages.")
        extracted = []
        length = 0
        for page in reader.pages:
            part = page.extract_text() or ""
            length += len(part) + 1
            if length > MAX_TEXT + 1:
                raise ValueError("The extracted text exceeds 40,000 characters. Paste the relevant section.")
            extracted.append(part)
        return "\n".join(extracted)
    except (PyPdfError, KeyError, TypeError, OverflowError, OSError) as error:
        raise ValueError("The PDF could not be read. Try another file or paste the text.") from error


def extract_docx(body):
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            document = archive.getinfo("word/document.xml")
            if document.file_size > MAX_DOCX_XML:
                raise ValueError("The DOCX is too large to extract. Paste the relevant section.")
            xml = archive.read(document)
        if re.search(rb"<!\s*(?:DOCTYPE|ENTITY)\b", xml, flags=re.I):
            raise ValueError("The DOCX contains unsupported XML declarations.")
        root = ElementTree.fromstring(xml)
        return "\n".join(
            "".join(node.text or "" for node in paragraph.iter(WORD_NAMESPACE + "t"))
            for paragraph in root.iter(WORD_NAMESPACE + "p")
        )
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError, RuntimeError, NotImplementedError) as error:
        raise ValueError("The DOCX could not be read. Try another file or paste the text.") from error


def extract_document(filename, body):
    extension = Path(filename).suffix.casefold()
    if extension == ".pdf":
        text = extract_pdf(body)
    elif extension == ".docx":
        text = extract_docx(body)
    elif extension == ".txt":
        try:
            encoding = "utf-16" if body.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
            text = body.decode(encoding)
        except UnicodeDecodeError as error:
            raise ValueError("Save the TXT as UTF-8 or UTF-16 and try again.") from error
    else:
        raise ValueError("Choose a TXT, PDF or DOCX file.")
    if not text.strip():
        raise ValueError("No readable text was found. For a scanned PDF, run OCR or paste the text.")
    return text


class LedgerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_content(self, status, mime, body):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'",
        )
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, data):
        self.send_content(status, "application/json; charset=utf-8", json.dumps(data).encode())

    def do_GET(self):
        route = urlsplit(self.path).path
        if route == "/api/model":
            self.send_json(200, self.server.model_info)
        elif route in STATIC:
            file, mime = STATIC[route]
            self.send_content(200, mime, (ROOT / "web" / file).read_bytes())
        else:
            self.send_json(404, {"error": "Page not found."})

    def do_POST(self):
        if urlsplit(self.path).path != "/api/classify":
            self.close_connection = True
            self.send_json(404, {"error": "Page not found."})
            return
        if not self.allowed_origin():
            self.close_connection = True
            self.send_json(403, {"error": "Open Ledger at its localhost address."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("Add a complaint before classifying.")
        except ValueError:
            self.close_connection = True
            self.send_json(400, {"error": "Add a complaint before classifying."})
            return
        if length > MAX_FILE:
            self.close_connection = True
            self.send_json(413, {"error": "Choose a file smaller than 8 MB."})
            return

        content_type = self.headers.get_content_type()
        body = self.rfile.read(length)
        try:
            if content_type == "application/octet-stream":
                filename = unquote(self.headers.get("X-Filename", ""))
                text = extract_document(filename, body)
            elif content_type == "application/json":
                payload = json.loads(body)
                if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
                    raise ValueError("Provide the complaint as text.")
                text = payload["text"]
            else:
                self.send_json(415, {"error": "Send text or a supported document."})
                return
            if len(text) > MAX_TEXT:
                raise ValueError("Use up to 40,000 characters. Paste the relevant section.")
            result = self.server.classifier.predict(text)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json(400, {"error": "The submitted text could not be read."})
        except ValueError as error:
            self.send_json(422, {"error": str(error)})
        else:
            self.send_json(200, result)

    def allowed_origin(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        return self.headers.get("Host") in hosts and (
            origin is None or origin in {f"http://{host}" for host in hosts}
        )


def serve(port, open_browser=False):
    classifier = Classifier()
    report = json.loads((ROOT / "analysis" / "metrics.json").read_text(encoding="utf-8"))
    with ThreadingHTTPServer(("127.0.0.1", port), LedgerHandler) as server:
        server.classifier = classifier
        server.model_info = {
            "model": ROUTER_NAME,
            "labels": LABELS,
            "test_count": report["data"]["splits"]["test"],
            "macro_f1": report["ensemble"]["macro_f1"],
        }
        address = f"http://127.0.0.1:{server.server_port}"
        print(f"Ledger is ready at {address}", flush=True)
        if open_browser:
            webbrowser.open(address)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open Ledger in the default browser.")
    args = parser.parse_args()
    try:
        serve(args.port, args.open)
    except FileNotFoundError as error:
        parser.exit(1, f"Missing project file: {error.filename}. Extract the complete project before running Ledger.\n")
