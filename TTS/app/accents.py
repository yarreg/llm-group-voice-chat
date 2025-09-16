import re
import unicodedata
from typing import Dict, Optional, Tuple
import ruaccent

from .config import get_config
from .logging import get_logger

logger = get_logger(__name__)


class AccentError(Exception):
    """Custom exception for accent-related errors"""
    pass


class AccentProcessor:
    """Handles Russian accent processing with ruaccent and manual overrides"""

    def __init__(self):
        self.config = get_config()
        self.overrides = self.config.accent.overrides
        self.ruaccent_processor = None

        if self.config.accent.enabled:
            try:
                self.ruaccent_processor = ruaccent.Accenter()
                logger.info("RuAccent processor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize RuAccent: {e}")
                if self.config.f5_tts.fail_fast:
                    raise AccentError(f"Failed to initialize RuAccent: {e}")
                self.ruaccent_processor = None
        else:
            logger.info("Accent processing disabled")

    def normalize_text(self, text: str) -> str:
        """Normalize text for processing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        # Normalize dashes
        text = text.replace('–', '-').replace('—', '-')

        # Normalize other punctuation
        text = text.replace('...', '…')

        return text

    def apply_manual_overrides(self, text: str) -> str:
        """Apply manual accent overrides"""
        if not self.overrides:
            return text

        processed_text = text
        for original, accented in self.overrides.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            processed_text = pattern.sub(accented, processed_text)

        if processed_text != text:
            logger.info(f"Applied manual accent overrides: {text} -> {processed_text}")

        return processed_text

    def convert_plus_to_acute(self, text: str) -> str:
        """Convert + accent notation to U+0301 acute accent"""
        # Find words with + accent marks (e.g., "р+еке")
        words = text.split()
        processed_words = []

        for word in words:
            if '+' in word:
                # Find the + character and replace it with acute accent on previous character
                plus_pos = word.find('+')
                if plus_pos > 0:
                    # Insert acute accent before the + character
                    accented_word = word[:plus_pos] + '\u0301' + word[plus_pos+1:]
                    processed_words.append(accented_word)
                else:
                    processed_words.append(word)
            else:
                processed_words.append(word)

        return ' '.join(processed_words)

    def convert_acute_to_plus(self, text: str) -> str:
        """Convert U+0301 acute accent to + notation for logging"""
        # Replace U+0301 with + before the character
        return text.replace('\u0301', '+')

    def apply_ruaccent(self, text: str) -> str:
        """Apply ruaccent processing"""
        if not self.ruaccent_processor:
            logger.warning("RuAccent processor not available")
            return text

        try:
            # Process with ruaccent
            accented_text = self.ruaccent_processor.process(text)

            # Convert + notation to acute accent
            accented_text = self.convert_plus_to_acute(accented_text)

            logger.info(f"RuAccent processing: {text} -> {self.convert_acute_to_plus(accented_text)}")
            return accented_text

        except Exception as e:
            logger.error(f"RuAccent processing failed: {e}")
            return text

    def process_text(self, text: str) -> str:
        """Main text processing pipeline"""
        if not self.config.accent.enabled:
            logger.debug("Accent processing disabled, returning original text")
            return text

        logger.info(f"Processing text for accents: {text[:100]}...")

        # Step 1: Normalize text
        normalized_text = self.normalize_text(text)
        logger.debug(f"Normalized text: {normalized_text}")

        # Step 2: Apply manual overrides (highest priority)
        override_text = self.apply_manual_overrides(normalized_text)
        logger.debug(f"After manual overrides: {self.convert_acute_to_plus(override_text)}")

        # Step 3: Apply ruaccent
        accented_text = self.apply_ruaccent(override_text)
        logger.debug(f"After ruaccent: {self.convert_acute_to_plus(accented_text)}")

        # Final validation
        if not accented_text.strip():
            raise AccentError("Text became empty after accent processing")

        logger.info(f"Final accented text: {self.convert_acute_to_plus(accented_text[:100])}...")
        return accented_text

    def has_accent_marks(self, text: str) -> bool:
        """Check if text contains accent marks"""
        return '\u0301' in text or '+' in text

    def remove_accent_marks(self, text: str) -> str:
        """Remove accent marks from text"""
        # Remove acute accents
        text = unicodedata.normalize('NFD', text)
        text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        text = unicodedata.normalize('NFC', text)

        # Remove + accent marks
        text = re.sub(r'\+(\w)', r'\1', text)

        return text

    def get_accent_positions(self, text: str) -> Dict[int, str]:
        """Get positions of accent marks in text"""
        positions = {}

        for i, char in enumerate(text):
            if char == '\u0301':
                positions[i-1] = 'acute'  # Acute accent on previous character
            elif char == '+':
                # Check if it's used as accent mark
                if i > 0 and i < len(text) - 1 and text[i-1].isalpha() and text[i+1].isalpha():
                    positions[i-1] = 'plus'

        return positions

    def validate_accented_text(self, text: str) -> Tuple[bool, Optional[str]]:
        """Validate that accented text is properly formatted"""
        try:
            # Check for unmatched + characters
            plus_count = text.count('+')
            if plus_count > 0:
                # Count valid + accents (between letters)
                valid_plus = len(re.findall(r'\w\+\w', text))
                if plus_count != valid_plus:
                    return False, "Invalid + accent placement"

            # Check for unmatched acute accents
            acute_count = text.count('\u0301')
            if acute_count > 0:
                # Check if acute accents are on letters
                for i, char in enumerate(text):
                    if char == '\u0301' and (i == 0 or not text[i-1].isalpha()):
                        return False, "Invalid acute accent placement"

            return True, None

        except Exception as e:
            return False, f"Validation error: {e}"


# Global accent processor instance
_accent_processor: Optional[AccentProcessor] = None


def get_accent_processor() -> AccentProcessor:
    """Get global accent processor instance"""
    global _accent_processor
    if _accent_processor is None:
        _accent_processor = AccentProcessor()
    return _accent_processor


def process_text_accents(text: str) -> str:
    """Process text with accents (convenience function)"""
    processor = get_accent_processor()
    return processor.process_text(text)


def initialize_accents() -> None:
    """Initialize accent processing system"""
    logger.info("Initializing accent processing system...")

    try:
        # Create accent processor
        processor = get_accent_processor()

        # Test with sample text if enabled
        if processor.config.accent.enabled:
            test_text = "Привет, как дела?"
            processed = processor.process_text(test_text)
            logger.info(f"Accent processing test successful: {test_text} -> {processor.convert_acute_to_plus(processed)}")

        logger.info("Accent processing system initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize accent processing system: {e}")
        if get_config().f5_tts.fail_fast:
            raise
        else:
            logger.warning("Continuing without accent processing")