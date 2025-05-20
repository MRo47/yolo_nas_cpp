#include <algorithm>
#include <map>
#include <optional>
#include <string>

// Enum Mapping Helper class, Generic implementation for any enum type
template <typename EnumType>
class EnumMapping
{
private:
  std::map<EnumType, std::string> enum_to_str;
  std::map<std::string, EnumType> str_to_enum;

public:
  // Initialize with a list of mappings
  EnumMapping(std::initializer_list<std::pair<EnumType, std::string>> mappings)
  {
    for (const auto & [enum_val, str_val] : mappings) {
      enum_to_str[enum_val] = str_val;

      // Store both original and uppercase version for case-insensitive lookup
      str_to_enum[str_val] = enum_val;

      // Also store a version without display formatting for easier matching
      std::string clean_str = str_val;
      // Remove any text in parentheses
      size_t paren_pos = clean_str.find("(");
      if (paren_pos != std::string::npos) {
        clean_str = clean_str.substr(0, paren_pos);
        // Trim trailing spaces
        clean_str.erase(clean_str.find_last_not_of(" ") + 1);
      }
      str_to_enum[clean_str] = enum_val;
    }
  }

  // Convert enum to string
  std::string to_string(EnumType value) const
  {
    auto it = enum_to_str.find(value);
    if (it != enum_to_str.end()) {
      return it->second;
    }
    return "UNKNOWN";
  }

  // Convert string to enum
  std::optional<EnumType> from_string(const std::string & str) const
  {
    // Try direct lookup first
    auto it = str_to_enum.find(str);
    if (it != str_to_enum.end()) {
      return it->second;
    }

    // Try uppercase version for case-insensitive matching
    std::string upper_str = str;
    std::transform(upper_str.begin(), upper_str.end(), upper_str.begin(), [](unsigned char c) {
      return std::toupper(c);
    });

    for (const auto & [key, value] : str_to_enum) {
      std::string upper_key = key;
      std::transform(upper_key.begin(), upper_key.end(), upper_key.begin(), [](unsigned char c) {
        return std::toupper(c);
      });

      if (upper_key == upper_str) {
        return value;
      }
    }

    return std::nullopt;
  }
};